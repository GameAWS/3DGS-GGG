"""REAL 3DGS SINGLE-SCENE CONTROLLED-HOLE RENDERING VALIDATION.

Frozen primary methods: C0, C1-HARD, C3-SOFT.  Completion is unchanged.  Images are
produced exclusively by the repository's original CUDA Gaussian Grouping rasterizer.
"""
import argparse, csv, json, math, os, sys, time
from collections import defaultdict
from types import ModuleType, SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import-only compatibility shim.  The repository's ``scene`` package imports
# simple_knn while it is being initialised, although neither COLMAP loading nor
# inference-time rasterisation calls distCUDA2.  Keep the renderer untouched and
# make an accidental call fail loudly.
try:
    import simple_knn._C  # noqa: F401
except ImportError:
    simple_knn = ModuleType("simple_knn")
    simple_knn_c = ModuleType("simple_knn._C")
    def _unused_dist_cuda2(*_args, **_kwargs):
        raise RuntimeError("distCUDA2 is unavailable in inference-only rendering")
    simple_knn_c.distCUDA2 = _unused_dist_cuda2
    simple_knn._C = simple_knn_c
    sys.modules["simple_knn"] = simple_knn
    sys.modules["simple_knn._C"] = simple_knn_c
from completion import geometry
from completion.gaussian_model import GaussianModel, get_projection_matrix, get_world_to_view_2
from completion.run_real_controlled import subset_model
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat
from utils.graphics_utils import focal2fov

METHODS = (("c0", "C0", "hard"), ("c1_hard", "C1", "hard"),
           ("c3_soft", "C3", "soft"))
GEOMETRY_KEYS = ("pred_to_gt", "gt_to_pred", "symmetric_chamfer", "fscore_2.0x",
                 "normal_error", "appearance_rmse", "seam_error")
IMAGE_KEYS = ("hole_psnr", "hole_ssim", "hole_lpips", "boundary_seam")


def frozen_geometry(rows_in):
    """Select only the pre-declared primary policies from the diagnostic table."""
    wanted={"C0":"hard","C1":"hard","C3":"soft"}
    return [r for r in rows_in if r.get("method") in wanted and
            r.get("policy",r.get("normal_affinity","hard")) == wanted[r["method"]]]


def rows(path):
    with open(path, newline="") as f: return list(csv.DictReader(f))


def write_csv(path, data):
    if not data: return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)


class RenderModel:
    """Shape-correct CUDA adapter; does not change any Gaussian value."""
    def __init__(self, model, device="cuda"):
        self.max_sh_degree = model.max_sh_degree
        self.active_sh_degree = model.active_sh_degree
        for name in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling",
                     "_rotation", "_objects_dc"):
            setattr(self, name, getattr(model, name).detach().to(device).contiguous())
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


class RealCamera:
    def __init__(self, uid, image_name, R, T, fovx, fovy, width, height):
        self.uid, self.image_name, self.R, self.T = uid, image_name, R, T
        self.FoVx, self.FoVy = fovx, fovy
        self.image_width, self.image_height = width, height
        self.world_view_transform = torch.tensor(get_world_to_view_2(R, T)).transpose(0, 1).cuda()
        self.projection_matrix = get_projection_matrix(.01, 100., fovx, fovy).transpose(0, 1).cuda()
        self.full_proj_transform = self.world_view_transform.unsqueeze(0).bmm(
            self.projection_matrix.unsqueeze(0)).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def load_cameras(data_root, max_width):
    sparse = os.path.join(data_root, "sparse", "0")
    extr = read_extrinsics_binary(os.path.join(sparse, "images.bin"))
    intr = read_intrinsics_binary(os.path.join(sparse, "cameras.bin"))
    result = []
    for image_id, item in sorted(extr.items(), key=lambda kv: kv[1].name):
        ci = intr[item.camera_id]
        scale = min(1., max_width / ci.width)
        width, height = int(round(ci.width*scale)), int(round(ci.height*scale))
        if ci.model == "SIMPLE_PINHOLE": fx = fy = ci.params[0]
        elif ci.model == "PINHOLE": fx, fy = ci.params[:2]
        else: raise RuntimeError("unsupported undistorted camera model " + ci.model)
        result.append(RealCamera(image_id, item.name, qvec2rotmat(item.qvec).T,
                                 np.asarray(item.tvec), focal2fov(fx, ci.width),
                                 focal2fov(fy, ci.height), width, height))
    return result


def camera_geometry(cam, center, radius, normal):
    Rwc = cam.R.T; point = Rwc @ center + cam.T
    if point[2] <= .01: return None
    fx = cam.image_width/(2*math.tan(cam.FoVx/2)); fy = cam.image_height/(2*math.tan(cam.FoVy/2))
    x = fx*point[0]/point[2] + cam.image_width/2
    y = fy*point[1]/point[2] + cam.image_height/2
    radius_px = max(fx, fy)*radius/point[2]
    margin = max(4., radius_px)
    if x < -margin or x >= cam.image_width+margin or y < -margin or y >= cam.image_height+margin or radius_px < 2:
        return None
    eye = np.linalg.inv(np.block([[Rwc, cam.T[:,None]], [np.zeros((1,3)), np.ones((1,1))]]))[:3,3]
    ray = eye-center; ray /= np.linalg.norm(ray)+1e-12
    frontal = abs(float(ray @ normal))
    return {"x":x,"y":y,"radius_px":radius_px,"depth":float(point[2]),"frontal":frontal}


def boundary_normal(model, center, radius):
    xyz = model.get_xyz.detach().cpu().numpy(); mask = np.linalg.norm(xyz-center, axis=1) <= radius
    kept = xyz[~mask]; bk, _ = geometry.detect_boundary_from_region(kept, center-radius, center+radius)
    ns = geometry.estimate_normals_local_pca_at(kept, bk, k=16)
    _, v = np.linalg.eigh(ns.T@ns); n = v[:,-1]; return n/(np.linalg.norm(n)+1e-12)


def select_cameras(cameras, center, radius, normal):
    visible = [(c, camera_geometry(c, center, radius, normal)) for c in cameras]
    visible = [(c,g) for c,g in visible if g]
    if not visible: raise RuntimeError("no real camera observes ROI")
    frontal = max(visible, key=lambda x:(x[1]["frontal"],x[1]["radius_px"]))
    if len(visible) == 1: return [("frontal",)+frontal]
    remain = [x for x in visible if x[0].uid != frontal[0].uid]
    oblique = min(remain, key=lambda x:abs(x[1]["frontal"]-.45))
    if len(visible) == 2: return [("frontal",)+frontal, ("oblique",)+oblique]
    used = {frontal[0].uid, oblique[0].uid}
    remain = [x for x in visible if x[0].uid not in used]
    nearby = max(remain, key=lambda x:(x[1]["radius_px"],-x[1]["depth"]))
    return [("frontal",)+frontal, ("oblique",)+oblique, ("nearby",)+nearby]


def render_cuda(model, cam, background):
    from gaussian_renderer import render
    pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    gpu = RenderModel(model)
    with torch.no_grad(): image = render(cam, gpu, pipe, background)["render"].clamp(0,1).cpu()
    del gpu; torch.cuda.empty_cache(); return image


def save_tensor(path, tensor):
    a=(tensor.permute(1,2,0).numpy()*255).round().clip(0,255).astype(np.uint8)
    Image.fromarray(a).save(path)


def region_mask(gt, hole):
    diff = torch.mean(torch.abs(gt-hole), dim=0).numpy()
    threshold = max(.01, float(np.percentile(diff, 95))*.12)
    core = diff > threshold
    if core.sum() < 16:
        threshold = max(.003, float(np.percentile(diff, 99)))
        core = diff >= threshold
    hole_mask = binary_dilation(core, iterations=2)
    outer = binary_dilation(hole_mask, iterations=8)
    boundary = outer & ~binary_dilation(hole_mask, iterations=1)
    return hole_mask, boundary, diff, threshold


def bbox(mask, pad=12):
    yy,xx=np.where(mask)
    if not len(xx): return (0,0,mask.shape[1],mask.shape[0])
    return (max(0,xx.min()-pad),max(0,yy.min()-pad),min(mask.shape[1],xx.max()+pad+1),min(mask.shape[0],yy.max()+pad+1))


def masked_psnr(pred, gt, mask):
    m=torch.from_numpy(mask).bool(); mse=torch.mean((pred[:,m]-gt[:,m])**2)
    return float(-10*torch.log10(mse.clamp_min(1e-12)))


def masked_ssim(pred, gt, mask):
    from skimage.metrics import structural_similarity
    x0,y0,x1,y1=bbox(mask,4); p=pred[:,y0:y1,x0:x1].permute(1,2,0).numpy(); g=gt[:,y0:y1,x0:x1].permute(1,2,0).numpy()
    local=mask[y0:y1,x0:x1]; p=np.where(local[...,None],p,g)
    return float(structural_similarity(g,p,data_range=1.,channel_axis=2))


def masked_lpips(pred, gt, mask, loss_fn):
    x0,y0,x1,y1=bbox(mask,8); p=pred[:,y0:y1,x0:x1].clone(); g=gt[:,y0:y1,x0:x1].clone()
    local=torch.from_numpy(mask[y0:y1,x0:x1]).bool(); p[:,~local]=g[:,~local]
    # AlexNet LPIPS needs a sufficiently large spatial support for all pooling
    # stages.  Resize both already-masked crops identically when the ROI is tiny.
    if min(p.shape[-2:]) < 64:
        size=(max(64,p.shape[-2]),max(64,p.shape[-1]))
        p=torch.nn.functional.interpolate(p[None],size=size,mode="bilinear",align_corners=False)[0]
        g=torch.nn.functional.interpolate(g[None],size=size,mode="bilinear",align_corners=False)[0]
    with torch.no_grad(): return float(loss_fn((p[None].cuda()*2-1),(g[None].cuda()*2-1)).item())


def image_metrics(pred, gt, hole_mask, band_mask, lpips_fn):
    full=np.ones_like(hole_mask,dtype=bool); out={}
    for name,mask in (("hole",hole_mask),("boundary",band_mask),("full",full)):
        out[name+"_psnr"]=masked_psnr(pred,gt,mask); out[name+"_ssim"]=masked_ssim(pred,gt,mask)
        out[name+"_lpips"]=masked_lpips(pred,gt,mask,lpips_fn)
    out["boundary_seam"]=float(torch.mean(torch.abs(pred[:,torch.from_numpy(band_mask)]-gt[:,torch.from_numpy(band_mask)])))
    return out


def panels(path, images, error_maps, mask):
    labels=list(images); x0,y0,x1,y1=bbox(mask,20); parts=[]
    for label in labels:
        a=(images[label].permute(1,2,0).numpy()*255).astype(np.uint8); im=Image.fromarray(a)
        canvas=Image.new("RGB",(im.width,im.height+24),"white"); canvas.paste(im,(0,24)); ImageDraw.Draw(canvas).text((5,5),label,fill="black"); parts.append(canvas)
    strip=Image.new("RGB",(sum(p.width for p in parts),parts[0].height),"white"); x=0
    for p in parts: strip.paste(p,(x,0)); x+=p.width
    strip.save(path)
    crops=[Image.fromarray((images[k][:,y0:y1,x0:x1].permute(1,2,0).numpy()*255).astype(np.uint8)).resize(((x1-x0)*3,(y1-y0)*3)) for k in labels]
    crop_strip=Image.new("RGB",(sum(p.width for p in crops),crops[0].height),"white"); x=0
    for p in crops: crop_strip.paste(p,(x,0)); x+=p.width
    crop_strip.save(os.path.join(os.path.dirname(path),"enlarged_crop.png"))
    # One shared error scale per view, so methods cannot look artificially similar
    # through independent normalisation.
    positive=np.concatenate([e.reshape(-1) for e in error_maps.values()])
    scale=max(float(np.percentile(positive,99.5)),1e-8)
    for k,e in error_maps.items():
        heat=(np.clip(e/scale,0,1)*255).astype(np.uint8); Image.fromarray(heat).save(os.path.join(os.path.dirname(path),"error_maps",k+".png"))


def select_geometric_cases(geom):
    """Rank ROIs before looking at renders (positive = Chamfer improvement)."""
    by=defaultdict(dict)
    for r in geom: by[r["roi"]][r["method"]]=r
    candidates=[]
    for roi,methods in by.items():
        if not all(m in methods for m in ("C0","C1","C3")): continue
        c0=float(methods["C0"]["symmetric_chamfer"])
        for method in ("C1","C3"):
            candidates.append({"roi":roi,"selected_method":method,
                               "c0_minus_method_chamfer":c0-float(methods[method]["symmetric_chamfer"])})
    def unique_cases(records):
        out=[]; seen=set()
        for r in records:
            if r["roi"] not in seen: out.append(r); seen.add(r["roi"])
            if len(out)==5: break
        return out
    improvements=unique_cases(sorted(candidates,key=lambda r:r["c0_minus_method_chamfer"],reverse=True))
    degradations=unique_cases(sorted(candidates,key=lambda r:r["c0_minus_method_chamfer"]))
    out=[]
    for rank,r in enumerate(improvements,1): out.append({"selection":"strongest_improvement","rank":rank,**r})
    for rank,r in enumerate(degradations,1): out.append({"selection":"strongest_degradation","rank":rank,**r})
    return out


def bh(p):
    p=np.asarray(p,dtype=float); q=np.full(len(p),np.nan); valid=np.isfinite(p)
    pv=p[valid]
    if not len(pv): return q
    order=np.argsort(pv); ranked=pv[order]*len(pv)/(np.arange(len(pv))+1)
    ranked=np.minimum.accumulate(ranked[::-1])[::-1]
    qv=np.empty(len(pv)); qv[order]=np.minimum(ranked,1); q[valid]=qv
    return q


def correlations(roi_rows, geom):
    gmap={(r["roi"],r["method"]):r for r in geom}; data=[]; ps=[]
    for gk in GEOMETRY_KEYS:
        for ik in IMAGE_KEYS:
            xs=[]; ys=[]
            for r in roi_rows:
                key=(r["roi"],r["geometry_method"])
                if key in gmap:
                    x,y=float(gmap[key][gk]),float(r[ik])
                    if np.isfinite(x) and np.isfinite(y): xs.append(x); ys.append(y)
            rho,p=spearmanr(xs,ys); data.append({"geometry_metric":gk,"image_metric":ik,"spearman_r":rho,"spearman_p":p,"n":len(xs)}); ps.append(p)
    qs=bh(ps)
    for r,q in zip(data,qs): r["spearman_q_fdr"]=q
    return data


def bootstrap(diff, n=10000):
    rng=np.random.default_rng(20260814); means=np.asarray([rng.choice(diff,len(diff),replace=True).mean() for _ in range(n)])
    return np.percentile(means,[2.5,97.5])


def paired(roi_rows):
    by=defaultdict(dict)
    for r in roi_rows: by[r["roi"]][r["method"]]=r
    out=[]
    for method in ("c1_hard","c3_soft"):
        for metric in IMAGE_KEYS:
            d=np.asarray([float(v[method][metric])-float(v["c0"][metric]) for v in by.values()])
            d=d[np.isfinite(d)]
            lo,hi=bootstrap(d)
            try: stat,p=wilcoxon(d)
            except ValueError: stat=p=float("nan")
            out.append({"comparison":method+"-c0","metric":metric,"n_rois":len(d),"mean_difference":d.mean(),"median_difference":np.median(d),"bootstrap_95_ci_low":lo,"bootstrap_95_ci_high":hi,"wilcoxon_stat":stat,"wilcoxon_p":p})
    return out


def plots(out, roi_rows, corr):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs(out,exist_ok=True); methods=("hole","c0","c1_hard","c3_soft")
    fig,axs=plt.subplots(1,3,figsize=(13,4))
    for ax,key in zip(axs,("hole_psnr","hole_ssim","hole_lpips")):
        ax.boxplot([[float(r[key]) for r in roi_rows if r["method"]==m] for m in methods],tick_labels=methods); ax.set_title(key)
    fig.tight_layout(); fig.savefig(os.path.join(out,"hole_metric_boxplots.png"),dpi=180); plt.close(fig)
    for y,name in (("hole_lpips","geometry_vs_lpips.png"),("hole_psnr","geometry_vs_psnr.png"),("boundary_seam","seam_geometry_vs_image.png")):
        fig,ax=plt.subplots(figsize=(6,5)); gm="seam_error" if y=="boundary_seam" else "symmetric_chamfer"
        gmap={(r["roi"],r["method"]):r for r in frozen_geometry(rows(args.geometry_csv))}
        for r in roi_rows:
            if r["method"]=="hole": continue
            gr=gmap.get((r["roi"],r["geometry_method"]));
            if gr and np.isfinite(float(gr[gm])) and np.isfinite(float(r[y])):
                ax.scatter(float(gr[gm]),float(r[y]),s=12,alpha=.6)
        ax.set(xlabel=gm,ylabel=y); fig.tight_layout(); fig.savefig(os.path.join(out,name),dpi=180); plt.close(fig)


def main():
    global args
    ap=argparse.ArgumentParser(); ap.add_argument("--checkpoint",required=True); ap.add_argument("--data",required=True); ap.add_argument("--rois-csv",required=True); ap.add_argument("--geometry-csv",required=True); ap.add_argument("--out",required=True); ap.add_argument("--max-width",type=int,default=800); args=ap.parse_args()
    os.makedirs(args.out,exist_ok=True); model=GaussianModel(3); model.load_ply(args.checkpoint); xyz=model.get_xyz.detach().cpu().numpy()
    cams=load_cameras(args.data,args.max_width); bg=torch.zeros(3,device="cuda"); import lpips; lpips_fn=lpips.LPIPS(net="alex").cuda().eval()
    descriptors=rows(args.rois_csv); camera_rows=[]; view_rows=[]
    for ri,d in enumerate(descriptors):
        roi=d["roi"]; center=np.asarray([d["center_x"],d["center_y"],d["center_z"]],float); radius=float(d["radius"]); mask=np.linalg.norm(xyz-center,axis=1)<=radius
        scene=SimpleNamespace(name=roi,model=model,center=center,roi_center=center,roi_radius=radius,hole_lo=center-radius,hole_hi=center+radius)
        normal=boundary_normal(model,center,radius); selected=select_cameras(cams,center,radius,normal); hole_model=subset_model(model,~mask)
        completed={}
        for key,variant,aff in METHODS:
            result=geometry.run_completion(model,scene,baseline=variant,seed=0,normal_affinity=aff,semantic_gate="hard",hole_mask_override=mask,spawn_rule="count_matched")
            completed[key]=geometry.append_gaussians(hole_model,result)
        for role,cam,g in selected:
            camera_rows.append({"roi":roi,"camera_role":role,"camera_id":cam.uid,"image_name":cam.image_name,**g})
            cdir=os.path.join(args.out,"renders",roi,str(cam.uid)); os.makedirs(os.path.join(cdir,"error_maps"),exist_ok=True)
            ims={"gt":render_cuda(model,cam,bg),"hole":render_cuda(hole_model,cam,bg)}
            for key,_,_ in METHODS: ims[key]=render_cuda(completed[key],cam,bg)
            hm,bm,_,thr=region_mask(ims["gt"],ims["hole"])
            for key,im in ims.items(): save_tensor(os.path.join(cdir,key+".png"),im)
            errors={k:torch.mean(torch.abs(v-ims["gt"]),0).numpy() for k,v in ims.items() if k!="gt"}; panels(os.path.join(cdir,"comparison.png"),ims,errors,hm)
            for key,im in ims.items():
                if key=="gt": continue
                met=image_metrics(im,ims["gt"],hm,bm,lpips_fn); geometry_method={"c0":"C0","c1_hard":"C1","c3_soft":"C3"}.get(key,"")
                view_rows.append({"roi":roi,"camera_role":role,"camera_id":cam.uid,"image_name":cam.image_name,"method":key,"geometry_method":geometry_method,"hole_pixels":int(hm.sum()),"boundary_pixels":int(bm.sum()),"mask_threshold":thr,**met})
        print("[render] {}/25 {}".format(ri+1,roi),flush=True)
    roi_rows=[]
    for (roi,method),group in sorted(defaultdict(list, {k:[r for r in view_rows if (r["roi"],r["method"])==k] for k in {(r["roi"],r["method"]) for r in view_rows}}).items()):
        row={"roi":roi,"method":method,"geometry_method":group[0]["geometry_method"],"n_cameras":len(group)}
        for key in [k for k in group[0] if k.endswith(("psnr","ssim","lpips","seam"))]:
            values=np.asarray([float(r[key]) for r in group],dtype=float)
            row[key]=float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
            row[key+"_n_valid"]=int(np.isfinite(values).sum())
        roi_rows.append(row)
    geom=frozen_geometry(rows(args.geometry_csv)); corr=correlations([r for r in roi_rows if r["method"] not in ("hole",)],geom); stats=paired(roi_rows)
    write_csv(os.path.join(args.out,"camera_selection.csv"),camera_rows); write_csv(os.path.join(args.out,"render_metrics_per_view.csv"),view_rows); write_csv(os.path.join(args.out,"render_metrics_per_roi.csv"),roi_rows); write_csv(os.path.join(args.out,"geometry_render_correlation.csv"),corr); write_csv(os.path.join(args.out,"paired_statistics.csv"),stats); write_csv(os.path.join(args.out,"case_selection.csv"),select_geometric_cases(geom)); plots(os.path.join(args.out,"plots"),roi_rows,corr)
    camera_count_distribution={str(n):sum(1 for d in descriptors if sum(r["roi"]==d["roi"] for r in camera_rows)==n) for n in (1,2,3)}
    meta={"label":"REAL 3DGS SINGLE-SCENE CONTROLLED-HOLE RENDERING VALIDATION","checkpoint":args.checkpoint,"data":args.data,"renderer":"original Gaussian Grouping diff_gaussian_rasterization CUDA","primary_methods":["GT","Hole","C0","C1-HARD","C3-SOFT"],"n_rois":len(descriptors),"requested_cameras_per_roi":3,"actual_camera_count_distribution":camera_count_distribution,"total_camera_views":len(camera_rows),"resolution_max_width":args.max_width,"gt_used_for_completion":False,"camera_selection_uses_removed_gt":False}
    json.dump(meta,open(os.path.join(args.out,"metadata.json"),"w"),indent=2)
    best={m:{k:float(np.nanmean([float(r[k]) for r in roi_rows if r["method"]==m])) for k in ("hole_psnr","hole_ssim","hole_lpips")} for m in ("c0","c1_hard","c3_soft")}
    report=["# REAL 3DGS SINGLE-SCENE CONTROLLED-HOLE RENDERING VALIDATION","","Controlled removal of Gaussians that originally existed in ramen; not object removal and not unseen-region hallucination.","","## Frozen protocol","","- C1-HARD and C3-SOFT are global policies fixed before rendering.","- Original Gaussian Grouping CUDA rasterizer and real COLMAP cameras only.","- Three GT-independent geometrically selected cameras per ROI; metrics aggregated within ROI before tests.","- Held-out Gaussians are used only for GT rendering and evaluation masks.","","## Aggregate hole metrics","",json.dumps(best,indent=2),"","## Paired statistics","",json.dumps(stats,indent=2),"","## Final answers","","See numeric CSVs and rendered failure panels. Conclusions must be based on completed real renders only."]
    open(os.path.join(args.out,"validation_report.md"),"w").write("\n".join(report)+"\n")
    print("[rendering-validation] done",flush=True)

if __name__=="__main__": main()
