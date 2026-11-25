"""Test source position, relative registration and ePSF sensitivity on the F200W epochs."""

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
from scipy.linalg import cho_solve, cho_factor
from scipy.special import logsumexp

from scripts.validate_m10_epochs import EPOCHS, RA_DEC, epoch_inputs, positions
from scripts.validate_m10_joint import CLUMPS, frame_inputs, host_vector
from scripts.validate_m10_target import adopted_alignment


LATE = EPOCHS[-2:]


def templates(frames, host, alternate):
    result = {}
    for epoch, frame in frames.items():
        op = frame["op"]
        if alternate and epoch in LATE:
            op = frames[LATE[1-LATE.index(epoch)]]["op"]
        geometry = np.array(list(host["geometry"].values()))
        frame = {**frame, "op":op}
        native = frame["native_offset"]
        context = frame["context"]

        def scene(delta):
            projected = {**frame, "native_offset":native+delta}
            basis = [host_vector(projected,geometry)]
            for name in CLUMPS:
                local = native+delta+frame["clump_offsets"][name]
                basis.append(op.render_point(*local,context))
            return np.array(basis)

        static = scene(np.zeros(2))
        amps = [host["host"]["value"],*(host["clumps"][name]["value"] for name in CLUMPS)]
        i = LATE.index(epoch) if epoch in LATE else None
        derivatives = []
        for axis in np.eye(2):
            step = .04*axis
            deriv = np.asarray(amps)@(scene(step)-scene(-step)).reshape(len(amps),-1)/.08
            if i is not None:
                flux = host["point"][epoch]["value"]
                plus = op.render_point(*(native+step),context)
                minus = op.render_point(*(native-step),context)
                deriv += flux*(plus-minus).ravel()/.08
            derivatives.append(frame["whiten"](deriv.reshape(context.shape)))
        result[epoch] = (frame, static, np.column_stack(derivatives))
    return result


def evaluate(prepared, registration_sigma, offsets, fixed_position):
    n = next(iter(prepared.values()))[0]["data"].size
    data, baselines, dots, weight = [], [], [], []
    count = len(EPOCHS)+1+len(CLUMPS)+len(LATE)
    for i,epoch in enumerate(EPOCHS):
        frame, static, gradient = prepared[epoch]
        stamp = slice(i*n,(i+1)*n)
        local = np.zeros((n,count))
        local[:, :len(CLUMPS)+1] = np.column_stack([frame["whiten"](image) for image in static])
        local[:,len(CLUMPS)+1+i] = frame["sky"]
        # In whitened pixel space, independent Gaussian registration shifts
        # add a rank-two covariance on each direct epoch.
        grad = registration_sigma*gradient
        factor = cho_factor(np.eye(2)+grad.T@grad)
        def whiten_precision(vector, grad=grad, factor=factor):
            return vector - grad@cho_solve(factor,grad.T@vector)
        data.append(frame["data"])
        baselines.append(local)
        weight.append(whiten_precision)
        if epoch in LATE:
            dots.append((stamp, frame))
    data = np.concatenate(data)
    baseline = np.vstack(baselines)

    def precision(matrix):
        blocks = np.split(matrix,len(EPOCHS))
        return np.concatenate([apply(block) for apply,block in zip(weight,blocks)])

    powered_data = precision(data)
    baseline_power = precision(baseline)
    sample = []
    positions_to_try = [(0.,0.)] if fixed_position else [(dx,dy) for dx in offsets for dy in offsets]
    for dx,dy in positions_to_try:
        design = baseline.copy()
        whitened = baseline_power.copy()
        for i, (rows, frame) in enumerate(dots):
            image = frame["op"].render_point(*(frame["native_offset"]+[dx,dy]),frame["context"])
            dot = frame["whiten"](image)
            design[rows,count-2+i] = dot
            whitened[rows,count-2+i] = weight[len(EPOCHS)-2+i](dot)
        factor = cho_factor(design.T@whitened)
        beta_cov = cho_solve(factor,np.eye(count))
        beta = cho_solve(factor,design.T@powered_data)
        residual = data-design@beta
        chi2 = residual@precision(residual)
        # Integral over the linear amplitudes with common flat measures.
        log_weight = -.5*chi2-np.log(np.diag(factor[0])).sum()
        sample.append((dx,dy,chi2,log_weight,beta[-2:],beta_cov[-2:,-2:]))
    best = min(sample,key=lambda row:row[2])
    logs = np.array([row[3] for row in sample])
    probability = np.exp(logs-logsumexp(logs))
    means = np.array([row[4] for row in sample])
    covs = np.array([row[5] for row in sample])
    mean = probability@means
    centered = means-mean
    covariance = np.einsum("i,ijk->jk",probability,covs)+centered.T@(probability[:,None]*centered)
    edge = [max(abs(row[0]),abs(row[1])) == max(offsets) for row in sample]
    return dict(best_position_xy_px=list(best[:2]),best_chi2=best[2],best_flux_nJy=(1000*best[4]).tolist(),
                marginalized_flux_nJy=(1000*mean).tolist(),
                marginalized_sigma_nJy=(1000*np.sqrt(covariance.diagonal())).tolist(),
                position_rim_mass=float(probability[edge].sum()))


def main():
    parser = ArgumentParser(description=__doc__)
    for field in ("cutouts","early_psfs","products","baseline","output"):
        parser.add_argument(field,type=Path)
    args = parser.parse_args()
    host = json.loads(args.baseline.read_text())
    arrays, _, veto, wcs = epoch_inputs(args.cutouts,args.early_psfs,args.products/"PSFs")
    default, offsets = adopted_alignment(args.products,wcs,arrays["ep01"][0].shape)
    train,held = positions(arrays,veto,wcs)
    frames = frame_inputs(arrays,offsets,wcs,train,held,10)
    x,y = wcs.world_to_pixel_values(*RA_DEC)
    clump_offsets = {name:np.array(xy)-[1+x,1+y] for name,xy in CLUMPS.items()}
    for frame in frames.values():
        frame["clump_offsets"] = clump_offsets
    grid = np.linspace(-1.,1.,31)
    prepared = templates(frames,host,False)
    results = {"fixed_position":evaluate(prepared,0.,grid,True),
               "common_free_position":evaluate(prepared,0.,grid,False)}
    expected = [1000*host["point"][epoch]["value"] for epoch in LATE]
    np.testing.assert_allclose(results["fixed_position"]["best_flux_nJy"],expected,atol=1e-7)
    for jitter in (.15,.30):
        results[f"registration_{jitter:.2f}_px"] = evaluate(prepared,jitter,grid,False)
    results["swap_ep08_ep09_epsfs"] = evaluate(templates(frames,host,True),.15,grid,False)
    print(json.dumps(results,indent=2),flush=True)
    args.output.write_text(json.dumps(dict(
        registration_reference=default,grid_xy_px=grid.tolist(),baseline=str(args.baseline),
        assumptions="Host geometry fixed at wide-radius fit; host amplitudes and sky refit; Gaussian registration sensitivity is linearized and uses illustrative independent priors; swapped ePSFs are a stress test.",
        results=results),indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
