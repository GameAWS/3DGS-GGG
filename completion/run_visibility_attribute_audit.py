"""Visibility-aware controlled-hole benchmark and newborn attribute audit.

Geometry is frozen: C0, C1-HARD and C3-SOFT call the existing count-matched
completion implementation unchanged.  Benchmark construction uses only GT/Hole;
attribute variants reuse the exact saved newborn XYZ for their geometry method.
"""
import argparse, copy, json, math, os, sys, time
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from completion import geometry
from completion.run_real_controlled import subset_model
from completion import run_rendering_validation as rv

LABEL = "REAL 3DGS SINGLE-SCENE VISIBILITY-AWARE CONTROLLED-HOLE DIAGNOSTIC"
METHODS = (("C0", "hard"), ("C1", "hard"), ("C3", "soft"))
PRIMARY_ATTRIBUTES = ("A0_CURRENT", "A1_NEAREST", "A2_KNN", "A3_COMPONENT", "A4_SURFACE_AWARE")
FACTOR_ATTRIBUTES = ("F_OPACITY", "F_SCALE", "F_ROTATION", "F_SH")
ALL_ATTRIBUTES = PRIMARY_ATTRIBUTES + FACTOR_ATTRIBUTES

# Frozen before method evaluation; no C0/C1/C3 result enters these gates.
MIN_CONTRIBUTING_CAMERAS = 2
MIN_RAW_CHANGED_PIXELS = 64
MIN_MEAN_HOLE_LPIPS = 0.002
RAW_RGB_CHANGE_THRESHOLD = 0.01


def write_csv(path, data):
    rv.write_csv(path, data)


def load_image(path):
    return torch.from_numpy(np.asarray(Image.open(path), dtype=np.float32) / 255.0).permute(2, 0, 1)


def render_gpu(pc, camera, background):
    from gaussian_renderer import render
    pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    with torch.no_grad():
        return render(camera, pc, pipe, background)["render"].clamp(0, 1).cpu()


class MutableCompletedModel:
    """One GPU base allocation whose newborn rows are updated between variants."""
    def __init__(self, hole_model, n_new):
        self.max_sh_degree = hole_model.max_sh_degree
        self.active_sh_degree = hole_model.active_sh_degree
        self.n_new = n_new
        specs = {
            "_xyz": (n_new, 3), "_features_dc": (n_new, 1, 3),
            "_features_rest": (n_new,) + tuple(hole_model._features_rest.shape[1:]),
            "_opacity": (n_new, 1), "_scaling": (n_new, 3),
            "_rotation": (n_new, 4),
            "_objects_dc": (n_new,) + tuple(hole_model._objects_dc.shape[1:]),
        }
        for name, shape in specs.items():
            base = getattr(hole_model, name).detach().cuda()
            setattr(self, name, torch.cat([base, torch.zeros(shape, dtype=base.dtype, device="cuda")], 0).contiguous())

    def update(self, xyz, attrs):
        n = len(xyz)
        if n != self.n_new: raise RuntimeError("newborn count changed during attribute audit")
        values = {
            "_xyz": xyz, "_features_dc": np.asarray(attrs["features_dc"]).reshape(n, 1, 3),
            "_features_rest": np.asarray(attrs["features_rest"]).reshape(n, self._features_rest.shape[1], 3),
            "_opacity": attrs["opacity"], "_scaling": attrs["scaling"],
            "_rotation": attrs["rotation"],
            "_objects_dc": np.asarray(attrs["objects_dc"]).reshape(n, self._objects_dc.shape[1], 1),
        }
        with torch.no_grad():
            for name, value in values.items():
                getattr(self, name)[-n:].copy_(torch.as_tensor(value, dtype=getattr(self, name).dtype, device="cuda"))

    @property
    def get_xyz(self): return self._xyz
    @property
    def get_features(self): return torch.cat((self._features_dc, self._features_rest), dim=1)
    @property
    def get_objects(self): return self._objects_dc
    @property
    def get_opacity(self): return torch.sigmoid(self._opacity)
    @property
    def get_scaling(self): return torch.exp(self._scaling)
    @property
    def get_rotation(self): return torch.nn.functional.normalize(self._rotation)


def audit_metrics(pred, gt, hole_mask, boundary_mask, lpips_fn):
    out = {}
    for name, mask in (("hole", hole_mask), ("boundary", boundary_mask)):
        if not np.any(mask):
            out.update({name+"_psnr": np.nan, name+"_ssim": np.nan, name+"_lpips": np.nan})
        else:
            out[name+"_psnr"] = rv.masked_psnr(pred, gt, mask)
            out[name+"_ssim"] = rv.masked_ssim(pred, gt, mask)
            out[name+"_lpips"] = rv.masked_lpips(pred, gt, mask, lpips_fn)
    out["boundary_seam"] = (float(torch.mean(torch.abs(
        pred[:, torch.from_numpy(boundary_mask)] - gt[:, torch.from_numpy(boundary_mask)])))
        if np.any(boundary_mask) else np.nan)
    return out


def raw_contribution(gt, hole):
    diff = torch.mean(torch.abs(gt-hole), dim=0).numpy()
    changed = diff >= RAW_RGB_CHANGE_THRESHOLD
    return {"raw_changed_pixels": int(changed.sum()), "rgb_contribution_sum": float(diff.sum()),
            "maximum_pixel_contribution": float(diff.max()),
            "projected_changed_footprint": int(changed.sum())}


def boundary_data(model, mask, center, radius):
    xyz = model.get_xyz.detach().cpu().numpy(); kept_idx = np.where(~mask)[0]; kept = xyz[~mask]
    bk, spacing = geometry.detect_boundary_from_region(kept, center-radius, center+radius)
    boundary_idx = kept_idx[bk]
    return boundary_idx, xyz[boundary_idx], spacing


def camera_map(cameras): return {int(c.uid): c for c in cameras}


def construct_visibility_benchmark(model, cameras, descriptors, camera_rows, out, lpips_fn, bg):
    """GT/Hole-only benchmark construction; completion is never called here."""
    by_roi = defaultdict(list)
    cmap = camera_map(cameras)
    for row in camera_rows: by_roi[row["roi"]].append(cmap[int(row["camera_id"])])
    xyz = model.get_xyz.detach().cpu().numpy(); gt_gpu = rv.RenderModel(model)
    candidate_rows, view_rows, rendered_camera_ids = [], [], {}
    cache_root = os.path.join(out, "benchmark_construction")
    for index, d in enumerate(descriptors):
        roi=d["roi"]; center=np.asarray([d["center_x"],d["center_y"],d["center_z"]],float); radius=float(d["radius"])
        mask=np.linalg.norm(xyz-center,axis=1)<=radius; hole=subset_model(model,~mask); hole_gpu=rv.RenderModel(hole)
        valid_ids=[]; local=[]
        for cam in by_roi[roi]:
            gt=render_gpu(gt_gpu,cam,bg); empty=render_gpu(hole_gpu,cam,bg)
            hm,bm,_,threshold=rv.region_mask(gt,empty); met=audit_metrics(empty,gt,hm,bm,lpips_fn); contrib=raw_contribution(gt,empty)
            if contrib["raw_changed_pixels"] >= 16: valid_ids.append(int(cam.uid))
            row={"roi":roi,"camera_id":int(cam.uid),"image_name":cam.image_name,
                 "removed_gaussians":int(mask.sum()),"mask_threshold":threshold,**contrib,**met}
            local.append(row); view_rows.append(row)
            cdir=os.path.join(cache_root,roi,str(cam.uid));os.makedirs(cdir,exist_ok=True)
            rv.save_tensor(os.path.join(cdir,"gt.png"),gt);rv.save_tensor(os.path.join(cdir,"hole.png"),empty)
        finite_lp=[r["hole_lpips"] for r in local if np.isfinite(r["hole_lpips"])]
        summary={"roi":roi,"removed_gaussians":int(mask.sum()),"candidate_cameras":len(local),
                 "multi_view_visibility_count":sum(r["raw_changed_pixels"]>=16 for r in local),
                 "accumulated_image_contribution":sum(r["rgb_contribution_sum"] for r in local),
                 "maximum_pixel_contribution":max(r["maximum_pixel_contribution"] for r in local),
                 "maximum_projected_footprint":max(r["projected_changed_footprint"] for r in local),
                 "total_visible_pixels_influenced":sum(r["raw_changed_pixels"] for r in local),
                 "mean_hole_lpips":float(np.mean(finite_lp)) if finite_lp else np.nan,
                 "mean_hole_psnr":float(np.nanmean([r["hole_psnr"] for r in local])),
                 "mean_hole_ssim":float(np.nanmean([r["hole_ssim"] for r in local]))}
        summary["passes_visibility_gate"]=(summary["multi_view_visibility_count"]>=MIN_CONTRIBUTING_CAMERAS and
            summary["maximum_projected_footprint"]>=MIN_RAW_CHANGED_PIXELS and
            summary["mean_hole_lpips"]>=MIN_MEAN_HOLE_LPIPS)
        summary["gate"]=("visible_cameras>={}; max_changed_pixels>={}; mean_hole_lpips>={}".format(
            MIN_CONTRIBUTING_CAMERAS,MIN_RAW_CHANGED_PIXELS,MIN_MEAN_HOLE_LPIPS))
        candidate_rows.append(summary); rendered_camera_ids[roi]=valid_ids
        del hole_gpu; torch.cuda.empty_cache(); print("[benchmark] {}/{} {}".format(index+1,len(descriptors),roi),flush=True)
    selected=[]
    dmap={d["roi"]:d for d in descriptors}
    for row in candidate_rows:
        if row["passes_visibility_gate"]:
            selected.append({**dmap[row["roi"]],**row,"selected_camera_ids":";".join(map(str,rendered_camera_ids[row["roi"]]))})
    write_csv(os.path.join(out,"visibility_hole_candidates.csv"),candidate_rows)
    write_csv(os.path.join(out,"visibility_selected_rois.csv"),selected)
    write_csv(os.path.join(out,"gt_hole_render_metrics.csv"),view_rows)
    del gt_gpu;torch.cuda.empty_cache()
    return selected, rendered_camera_ids


def weighted_attributes(new_xyz, boundary_xyz, attrs, spacing, mode, comp_b=None, comp_n=None):
    tree=cKDTree(boundary_xyz); k=min(8,len(boundary_xyz)); d,idx=tree.query(new_xyz,k=k)
    if k==1: d=d[:,None];idx=idx[:,None]
    if mode=="nearest":
        weights=np.zeros_like(d,dtype=np.float32);weights[:,0]=1
    else:
        weights=1.0/(d+max(spacing*.25,1e-8))
        if mode=="component":
            same=comp_b[idx]==comp_n[:,None];weights*=same
            empty=weights.sum(1)==0;weights[empty]=0;weights[empty,0]=1
        weights/=weights.sum(1,keepdims=True)+1e-12
    return {key:np.sum(value[idx]*weights[:,:,None],axis=1).astype(np.float32) for key,value in attrs.items()},idx,weights


def surface_aware_attributes(result, boundary_xyz, attrs, component_attrs, neighbor_idx, neighbor_weights):
    n=len(result.new_xyz); out={k:v.copy() for k,v in component_attrs.items()}
    support_scales=np.exp(attrs["scaling"]); support_opacity=1/(1+np.exp(-attrs["opacity"][:,0]))
    scaling=np.empty((n,3),np.float32); rotations=np.empty((n,4),np.float32); opacities=np.empty((n,1),np.float32)
    matrices=[]
    for i,(normal,inds) in enumerate(zip(result.new_normals,neighbor_idx)):
        same_surface=neighbor_weights[i]>0
        inds=inds[same_surface] if np.any(same_surface) else inds[:1]
        normal=np.asarray(normal,float);normal/=np.linalg.norm(normal)+1e-12
        ref=np.array([1.,0.,0.]) if abs(normal[0])<.9 else np.array([0.,1.,0.])
        t1=np.cross(normal,ref);t1/=np.linalg.norm(t1)+1e-12;t2=np.cross(normal,t1)
        matrices.append(np.stack([t1,t2,normal],axis=1))
        local=np.sort(support_scales[inds],axis=1)
        tangent=min(float(np.median(local[:,-1])),float(result.boundary_spacing*.5))
        normal_scale=min(float(np.median(local[:,0])),tangent*.2)
        tangent=max(tangent,1e-6);normal_scale=max(normal_scale,1e-6)
        scaling[i]=np.log([tangent,tangent,normal_scale])
        opacity=float(np.percentile(support_opacity[inds],25));opacity=np.clip(opacity,.01,.5)
        opacities[i,0]=math.log(opacity/(1-opacity))
    q=Rotation.from_matrix(np.asarray(matrices)).as_quat() # xyzw -> wxyz
    rotations[:]=q[:,[3,0,1,2]]
    out["scaling"],out["rotation"],out["opacity"]=scaling,rotations,opacities
    return out


def attribute_variants(model, result):
    bidx=result.boundary_idx; bxyz=model.get_xyz.detach().cpu().numpy()[bidx]
    attrs=geometry._boundary_attr_arrays(model,bidx)
    a1,_,_=weighted_attributes(result.new_xyz,bxyz,attrs,result.boundary_spacing,"nearest")
    a2,_,_=weighted_attributes(result.new_xyz,bxyz,attrs,result.boundary_spacing,"knn")
    comp_b=result.component_labels if result.component_labels is not None else np.zeros(len(bidx),int)
    comp_n=result.surface_label if result.surface_label is not None else np.zeros(len(result.new_xyz),int)
    a3,idx,w3=weighted_attributes(result.new_xyz,bxyz,attrs,result.boundary_spacing,"component",comp_b,comp_n)
    a4=surface_aware_attributes(result,bxyz,attrs,a3,idx,w3)
    variants={"A0_CURRENT":{k:np.asarray(v).copy() for k,v in result.new_attributes.items()},
              "A1_NEAREST":a1,"A2_KNN":a2,"A3_COMPONENT":a3,"A4_SURFACE_AWARE":a4}
    for name,key in (("F_OPACITY","opacity"),("F_SCALE","scaling"),("F_ROTATION","rotation")):
        variants[name]={k:np.asarray(v).copy() for k,v in variants["A0_CURRENT"].items()};variants[name][key]=a4[key].copy()
    variants["F_SH"]={k:np.asarray(v).copy() for k,v in variants["A0_CURRENT"].items()}
    variants["F_SH"]["features_dc"]=a4["features_dc"].copy();variants["F_SH"]["features_rest"]=a4["features_rest"].copy()
    return variants,bxyz,attrs


def quat_matrices(wxyz):
    q=np.asarray(wxyz,dtype=float).copy();norm=np.linalg.norm(q,axis=1,keepdims=True);zero=norm[:,0]<1e-8
    q/=np.maximum(norm,1e-12);q[zero]=np.array([1.,0.,0.,0.]) # diagnostic-only fallback; rendering attrs remain untouched
    return Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()


def projected_radius(xyz, scale, cameras):
    result=np.zeros(len(xyz),np.float32)
    for cam in cameras:
        p=(cam.R.T@xyz.T).T+cam.T
        fx=cam.image_width/(2*math.tan(cam.FoVx/2));fy=cam.image_height/(2*math.tan(cam.FoVy/2))
        r=np.where(p[:,2]>.01,max(fx,fy)*np.max(scale,axis=1)/p[:,2],0)
        result=np.maximum(result,r)
    return result


def diagnostics(roi,method,variant,result,attrs,bxyz,battrs,cameras):
    scales=np.exp(attrs["scaling"]);opacity=1/(1+np.exp(-attrs["opacity"][:,0]));rot=quat_matrices(attrs["rotation"])
    tree=cKDTree(bxyz);_,nearest=tree.query(result.new_xyz,k=1)
    support_scale=np.exp(battrs["scaling"]);support_op=1/(1+np.exp(-battrs["opacity"][:,0]))
    median_scale=float(np.median(np.exp(np.mean(battrs["scaling"],axis=1))));median_op=float(np.median(support_op))
    radius=projected_radius(result.new_xyz,scales,cameras); rows=[]
    for i in range(len(scales)):
        cov=rot[i]@np.diag(scales[i]**2)@rot[i].T;n=result.new_normals[i]/(np.linalg.norm(result.new_normals[i])+1e-12)
        thick=float(np.sqrt(max(n@cov@n,0)));tangent=float(np.sqrt(max((np.trace(cov)-thick**2)/2,0)))
        axis=rot[i][:,int(np.argmin(scales[i]))];agreement=math.degrees(math.acos(np.clip(abs(axis@n),-1,1)))
        shdiff=float(np.linalg.norm(attrs["features_dc"][i]-battrs["features_dc"][nearest[i]]))
        rows.append({"roi":roi,"method":method,"attribute_variant":variant,"newborn_index":i,
                     "scale_x":scales[i,0],"scale_y":scales[i,1],"scale_z":scales[i,2],
                     "scale_relative_local_median":float(np.exp(np.mean(attrs["scaling"][i]))/max(median_scale,1e-12)),
                     "opacity":opacity[i],"opacity_relative_neighbor_median":opacity[i]/max(median_op,1e-12),
                     "covariance_eigenvalue_0":float(np.min(scales[i]**2)),"covariance_eigenvalue_1":float(np.median(scales[i]**2)),
                     "covariance_eigenvalue_2":float(np.max(scales[i]**2)),"normal_direction_thickness":thick,
                     "tangent_plane_footprint":tangent,"rotation_normal_error_deg":agreement,
                     "sh_dc_difference_nearest_support":shdiff,"projected_screen_radius_max":radius[i],
                     "maximum_alpha_contribution":opacity[i]})
    return rows


def aggregate_roi(view_rows):
    groups=defaultdict(list)
    for r in view_rows: groups[(r["roi"],r["method"],r["attribute_variant"])].append(r)
    out=[]; metric_keys=("hole_psnr","hole_ssim","hole_lpips","boundary_psnr","boundary_ssim","boundary_lpips","boundary_seam")
    for (roi,method,variant),group in sorted(groups.items()):
        row={"roi":roi,"method":method,"attribute_variant":variant,"n_cameras":len(group)}
        for key in metric_keys:
            a=np.asarray([float(x[key]) for x in group]);row[key]=float(np.nanmean(a)) if np.isfinite(a).any() else np.nan
            row[key+"_n_valid"]=int(np.isfinite(a).sum())
        out.append(row)
    return out


def paired_statistics(roi_rows):
    out=[]; by=defaultdict(dict)
    for r in roi_rows: by[(r["roi"],r["method"])][r["attribute_variant"]]=r
    metrics=("hole_psnr","hole_ssim","hole_lpips","boundary_seam")
    for method,_ in METHODS:
        for variant in ALL_ATTRIBUTES[1:]:
            for metric in metrics:
                d=[]
                for (roi,m),vals in by.items():
                    if m==method and "A0_CURRENT" in vals and variant in vals:
                        x=float(vals[variant][metric])-float(vals["A0_CURRENT"][metric])
                        if np.isfinite(x):d.append(x)
                a=np.asarray(d);lo,hi=rv.bootstrap(a) if len(a) else (np.nan,np.nan)
                try:stat,p=wilcoxon(a)
                except ValueError:stat=p=np.nan
                out.append({"method":method,"comparison":variant+"-A0_CURRENT","metric":metric,"n_rois":len(a),
                            "mean_difference":float(np.mean(a)),"median_difference":float(np.median(a)),
                            "bootstrap_95_ci_low":lo,"bootstrap_95_ci_high":hi,"wilcoxon_stat":stat,"wilcoxon_p":p})
    return out


def panels(out, selected, roi_rows, diag_rows):
    import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
    os.makedirs(os.path.join(out,"plots"),exist_ok=True)
    rows=[r for r in roi_rows if r["method"]!="HOLE"]
    best=min(ALL_ATTRIBUTES,key=lambda a:np.nanmean([r["hole_lpips"] for r in rows if r["attribute_variant"]==a]))
    method_best=min(((m,a) for m,_ in METHODS for a in ALL_ATTRIBUTES),key=lambda ma:np.nanmean([r["hole_lpips"] for r in rows if r["method"]==ma[0] and r["attribute_variant"]==ma[1]]))
    holes={r["roi"]:r for r in roi_rows if r["method"]=="HOLE"}
    bv={(r["roi"],r["method"]):r for r in rows if r["attribute_variant"]==best}
    changes=[]
    for roi in holes:
        available=[(r["hole_lpips"],method,r) for (name,method),r in bv.items() if name==roi]
        if available:
            _,method,r=min(available);changes.append((r["hole_lpips"]-holes[roi]["hole_lpips"],(roi,method)))
    representative=[x[1] for x in sorted(changes)[:3]+sorted(changes,reverse=True)[:3]]
    selected_map={r["roi"]:r for r in selected}
    for roi,method in representative:
        camera_ids=selected_map[roi]["selected_camera_ids"].split(";")
        def impact(cid):
            root=os.path.join(out,"renders",roi,cid);g=np.asarray(Image.open(os.path.join(root,"gt.png")),float);h=np.asarray(Image.open(os.path.join(root,"hole.png")),float)
            return float(np.abs(g-h).sum())
        cam_id=max(camera_ids,key=impact);cdir=os.path.join(out,"renders",roi,cam_id)
        paths=[os.path.join(cdir,"gt.png"),os.path.join(cdir,"hole.png"),os.path.join(cdir,method,"A0_CURRENT.png"),os.path.join(cdir,method,best+".png")]
        if not all(os.path.exists(p) for p in paths):continue
        ims=[Image.open(p).convert("RGB") for p in paths];labels=["GT","Hole",method+" Current",method+" "+best]
        canvas=Image.new("RGB",(sum(i.width for i in ims),ims[0].height+24),"white");x=0;draw=ImageDraw.Draw(canvas)
        for im,label in zip(ims,labels):canvas.paste(im,(x,24));draw.text((x+5,5),label,fill="black");x+=im.width
        canvas.save(os.path.join(cdir,"representative_comparison_"+method+".png"))
        gt=np.asarray(ims[0],float)/255;hm=np.abs(gt-np.asarray(ims[1],float)/255).mean(2)>=RAW_RGB_CHANGE_THRESHOLD
        x0,y0,x1,y1=rv.bbox(hm,20);crop=Image.new("RGB",(sum((x1-x0)*3 for _ in ims),(y1-y0)*3),"white");x=0
        for im in ims:
            q=im.crop((x0,y0,x1,y1)).resize(((x1-x0)*3,(y1-y0)*3));crop.paste(q,(x,0));x+=q.width
        crop.save(os.path.join(cdir,"representative_enlarged_crop_"+method+".png"))
        err=np.abs(gt-np.asarray(ims[-1],float)/255).mean(2);Image.fromarray((np.clip(err/max(np.percentile(err,99.5),1e-8),0,1)*255).astype(np.uint8)).save(os.path.join(cdir,"representative_error_map_"+method+".png"))
        dr=[d for d in diag_rows if d["roi"]==roi and d["method"]==method and d["attribute_variant"]==best]
        fig,ax=plt.subplots(figsize=(5,4));ax.scatter([d["projected_screen_radius_max"] for d in dr],[d["opacity"] for d in dr],s=12,alpha=.7)
        ax.set(xlabel="maximum projected radius (px)",ylabel="opacity",title=roi+" "+method+" "+best);fig.tight_layout();fig.savefig(os.path.join(cdir,"newborn_footprint_opacity_"+method+".png"),dpi=160);plt.close(fig)
    # Global diagnostic plots.
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for variant in PRIMARY_ATTRIBUTES:
        d=[x for x in diag_rows if x["attribute_variant"]==variant]
        axs[0].hist([x["projected_screen_radius_max"] for x in d],bins=40,alpha=.4,label=variant)
        axs[1].hist([x["opacity"] for x in d],bins=40,alpha=.4,label=variant)
    axs[0].set(xlabel="projected screen radius (px)",ylabel="newborn count");axs[1].set(xlabel="opacity");axs[1].legend(fontsize=7)
    fig.tight_layout();fig.savefig(os.path.join(out,"plots","newborn_footprint_opacity.png"),dpi=180);plt.close(fig)
    # Attribute performance and isolated-factor effect.
    fig,axs=plt.subplots(1,3,figsize=(14,4));labels=list(PRIMARY_ATTRIBUTES)
    for ax,key in zip(axs,("hole_psnr","hole_ssim","hole_lpips")):
        vals=[[r[key] for r in rows if r["attribute_variant"]==a] for a in labels]
        ax.boxplot(vals,tick_labels=[x.split("_")[0] for x in labels]);ax.set_title(key)
    fig.tight_layout();fig.savefig(os.path.join(out,"plots","attribute_ablation_hole_metrics.png"),dpi=180);plt.close(fig)
    factor_labels=["F_OPACITY","F_SCALE","F_ROTATION","F_SH"]
    a0={(r["roi"],r["method"]):r for r in rows if r["attribute_variant"]=="A0_CURRENT"}
    means=[]
    for factor in factor_labels:
        fr={(r["roi"],r["method"]):r for r in rows if r["attribute_variant"]==factor}
        means.append(np.mean([fr[k]["hole_lpips"]-v["hole_lpips"] for k,v in a0.items()]))
    fig,ax=plt.subplots(figsize=(7,4));ax.bar(factor_labels,means);ax.axhline(0,color="black",linewidth=.8);ax.set(ylabel="LPIPS change vs A0 (lower is better)",title="Isolated attribute replacement");fig.tight_layout();fig.savefig(os.path.join(out,"plots","single_factor_lpips_change.png"),dpi=180);plt.close(fig)
    cand=rv.rows(os.path.join(out,"visibility_hole_candidates.csv"));fig,ax=plt.subplots(figsize=(6,5))
    colors=["tab:blue" if str(r["passes_visibility_gate"]).lower()=="true" else "tab:gray" for r in cand]
    ax.scatter([float(r["maximum_projected_footprint"]) for r in cand],[float(r["mean_hole_lpips"]) for r in cand],c=colors)
    ax.axvline(MIN_RAW_CHANGED_PIXELS,color="black",linestyle="--");ax.axhline(MIN_MEAN_HOLE_LPIPS,color="black",linestyle="--");ax.set(xlabel="maximum changed pixels",ylabel="mean Hole LPIPS",title="GT/Hole-only visibility gate");fig.tight_layout();fig.savefig(os.path.join(out,"plots","visibility_candidate_gates.png"),dpi=180);plt.close(fig)
    return best,method_best,representative


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--checkpoint",required=True);ap.add_argument("--data",required=True)
    ap.add_argument("--rois-csv",required=True);ap.add_argument("--camera-csv",required=True);ap.add_argument("--out",required=True)
    ap.add_argument("--max-width",type=int,default=800);args=ap.parse_args();os.makedirs(args.out,exist_ok=True)
    model=rv.GaussianModel(3);model.load_ply(args.checkpoint);xyz=model.get_xyz.detach().cpu().numpy()
    cameras=rv.load_cameras(args.data,args.max_width);cmap=camera_map(cameras);descriptors=rv.rows(args.rois_csv);camera_rows=rv.rows(args.camera_csv)
    bg=torch.zeros(3,device="cuda");import lpips;lpips_fn=lpips.LPIPS(net="alex").cuda().eval()
    selected,valid_ids=construct_visibility_benchmark(model,cameras,descriptors,camera_rows,args.out,lpips_fn,bg)
    result_rows=[];diag_rows=[];geometry_manifest=[];gt_gpu=rv.RenderModel(model)
    for ri,d in enumerate(selected):
        roi=d["roi"];center=np.asarray([d["center_x"],d["center_y"],d["center_z"]],float);radius=float(d["radius"]);mask=np.linalg.norm(xyz-center,axis=1)<=radius
        cams=[cmap[i] for i in valid_ids[roi]];hole=subset_model(model,~mask);hole_gpu=rv.RenderModel(hole)
        refs={}
        for cam in cams:
            gt=render_gpu(gt_gpu,cam,bg);empty=render_gpu(hole_gpu,cam,bg);hm,bm,_,_=rv.region_mask(gt,empty);refs[int(cam.uid)]=(gt,empty,hm,bm)
            cdir=os.path.join(args.out,"renders",roi,str(cam.uid));os.makedirs(cdir,exist_ok=True);rv.save_tensor(os.path.join(cdir,"gt.png"),gt);rv.save_tensor(os.path.join(cdir,"hole.png"),empty)
            met=audit_metrics(empty,gt,hm,bm,lpips_fn);result_rows.append({"roi":roi,"camera_id":int(cam.uid),"method":"HOLE","attribute_variant":"HOLE",**met})
        scene=SimpleNamespace(name=roi,model=model,center=center,roi_center=center,roi_radius=radius,hole_lo=center-radius,hole_hi=center+radius)
        method_results={}
        for method,affinity in METHODS:
            result=geometry.run_completion(model,scene,baseline=method,seed=0,normal_affinity=affinity,semantic_gate="hard",hole_mask_override=mask,spawn_rule="count_matched")
            method_results[method]=result;gdir=os.path.join(args.out,"frozen_xyz",roi);os.makedirs(gdir,exist_ok=True)
            np.save(os.path.join(gdir,method+".npy"),result.new_xyz)
            geometry_manifest.append({"roi":roi,"method":method,"normal_affinity":affinity,"n_spawn":len(result.new_xyz),"xyz_file":os.path.join("frozen_xyz",roi,method+".npy")})
        counts={len(r.new_xyz) for r in method_results.values()}
        if len(counts)!=1:raise RuntimeError("count-matched geometry changed across methods")
        mutable=MutableCompletedModel(hole,counts.pop())
        for method,_ in METHODS:
            result=method_results[method];variants,bxyz,battrs=attribute_variants(model,result)
            for variant,attrs in variants.items():
                diag_rows.extend(diagnostics(roi,method,variant,result,attrs,bxyz,battrs,cams));mutable.update(result.new_xyz,attrs)
                for cam in cams:
                    image=render_gpu(mutable,cam,bg);gt,empty,hm,bm=refs[int(cam.uid)];met=audit_metrics(image,gt,hm,bm,lpips_fn)
                    result_rows.append({"roi":roi,"camera_id":int(cam.uid),"method":method,"attribute_variant":variant,**met})
                    if variant in PRIMARY_ATTRIBUTES:
                        cdir=os.path.join(args.out,"renders",roi,str(cam.uid),method);os.makedirs(cdir,exist_ok=True);rv.save_tensor(os.path.join(cdir,variant+".png"),image)
        del hole_gpu,mutable;torch.cuda.empty_cache();print("[attribute] {}/{} {}".format(ri+1,len(selected),roi),flush=True)
    roi_rows=aggregate_roi(result_rows);stats=paired_statistics(roi_rows)
    write_csv(os.path.join(args.out,"frozen_xyz_manifest.csv"),geometry_manifest);write_csv(os.path.join(args.out,"newborn_attribute_diagnostics.csv"),diag_rows)
    write_csv(os.path.join(args.out,"attribute_ablation_results.csv"),result_rows);write_csv(os.path.join(args.out,"per_roi_render_metrics.csv"),roi_rows);write_csv(os.path.join(args.out,"paired_statistics.csv"),stats)
    best,method_best,representative=panels(args.out,selected,roi_rows,diag_rows)
    meta={"label":LABEL,"checkpoint":args.checkpoint,"data":args.data,"benchmark_construction":{
        "candidate_rois":len(descriptors),"selected_rois":len(selected),"gates":{"minimum_contributing_cameras":MIN_CONTRIBUTING_CAMERAS,
        "minimum_raw_changed_pixels":MIN_RAW_CHANGED_PIXELS,"raw_rgb_change_threshold":RAW_RGB_CHANGE_THRESHOLD,"minimum_mean_hole_lpips":MIN_MEAN_HOLE_LPIPS}},
        "geometry_methods":[m for m,_ in METHODS],"attribute_variants":list(ALL_ATTRIBUTES),"gt_used_for_completion":False,
        "xyz_changed_across_attribute_variants":False,"best_attribute_strategy_by_lpips":best,"best_method_attribute_by_lpips":method_best,
        "representative_pairs":[list(x) for x in representative]}
    with open(os.path.join(args.out,"metadata.json"),"w") as f:json.dump(meta,f,indent=2)
    hole={r["roi"]:r for r in roi_rows if r["method"]=="HOLE"};completion=[r for r in roi_rows if r["method"]!="HOLE"]
    attr_mean={a:float(np.mean([r["hole_lpips"] for r in completion if r["attribute_variant"]==a])) for a in ALL_ATTRIBUTES}
    factor_best=min(FACTOR_ATTRIBUTES,key=lambda a:attr_mean[a]);a4={m:{} for m,_ in METHODS};failures=[]
    for method,_ in METHODS:
        for r in completion:
            if r["method"]==method and r["attribute_variant"]=="A4_SURFACE_AWARE":a4[method][r["roi"]]=r
    for roi in hole:
        if all(a4[m][roi]["hole_lpips"]>hole[roi]["hole_lpips"] for m,_ in METHODS):failures.append(roi)
    lines=["# "+LABEL,"",
        "1. **Original visually trivial/insufficient holes:** {}/{}; {} passed the frozen GT/Hole-only gate.".format(len(descriptors)-len(selected),len(descriptors),len(selected)),"",
        "2. **Visibility-aware holes:** yes; selected Hole mean LPIPS is {:.5f}.".format(np.mean([r["hole_lpips"] for r in hole.values()])),"",
        "3. **Fixed-XYZ attribute effect:** A0 LPIPS {:.5f} -> A4 LPIPS {:.5f}.".format(attr_mean["A0_CURRENT"],attr_mean["A4_SURFACE_AWARE"]),"",
        "4. **Dominant isolated factor:** {} (LPIPS {:.5f}); rotation-only LPIPS {:.5f}.".format(factor_best,attr_mean[factor_best],attr_mean["F_ROTATION"]),"",
        "5. **Best strategy:** {}; best method/strategy is {} + {}.".format(best,method_best[0],method_best[1]),""]
    lp_parts=[];all_parts=[]
    for method,_ in METHODS:
        vals=a4[method];lp=sum(vals[r]["hole_lpips"]<hole[r]["hole_lpips"] for r in vals);all3=sum(vals[r]["hole_lpips"]<hole[r]["hole_lpips"] and vals[r]["hole_psnr"]>hole[r]["hole_psnr"] and vals[r]["hole_ssim"]>hole[r]["hole_ssim"] for r in vals)
        lp_parts.append("{} {}/{}".format(method,lp,len(vals)));all_parts.append("{} {}/{}".format(method,all3,len(vals)))
    lines += ["6. **Majority LPIPS improvement over Hole:** {}.".format("; ".join(lp_parts)),"",
              "7. **Simultaneous PSNR/SSIM/LPIPS improvement:** {}.".format("; ".join(all_parts)),"",
              "8. **Failures after A4:** {}.".format(", ".join(failures)),"",
              "9. **Object-move readiness:** not yet; validate the frozen initializer across multiple scenes and larger disocclusions.",""]
    with open(os.path.join(args.out,"validation_report.md"),"w") as f:f.write("\n".join(lines))
    print("[audit] complete selected={} best={} method_best={}".format(len(selected),best,method_best),flush=True)


if __name__=="__main__":main()
