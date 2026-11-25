# ADR 001: PSF response and units

Status: accepted for the new API design; real PSF response is an open blocker.

The two example PSFs are finite 535×535 arrays. README and notebook callers
use oversampling=4. This is evidence of intended usage, not proof of the
generation method or sampling semantics. Full headers, sums, hashes and actual
downsampling losses are in `psf-audit.json`. The repository supplies no PSF
generation script establishing star selection/reconstruction, ePSF versus
pre-pixel intensity, included pixel response, drizzle kernel/pixfrac/scale,
reference center, or infinite-aperture normalization. These remain **unknown**.
Do not infer those facts from a filename, image appearance, or missing keywords.
Before real-data rendering is validated, obtain the generation recipe and input
stars/exposures, reconstruction software/version, response and drizzle settings,
center convention, and calibration/aperture reference from the PSF producer.

New images use explicitly declared flux per output pixel; errors use the same
unit. Surface brightness input (e.g. MJy/sr) needs a separate pixel solid-angle
conversion of both science and error arrays. No implicit unit inference.
Native centers are integer indices, array order is [y,x], offsets increase
column/row indices. The stamp center is ((nx-1)/2,(ny-1)/2); a subpixel center
in native pixel j is j+(a+0.5)/s-0.5. PSF origin must be explicit, not recomputed
with argmax. PA increases from +x to +y; re is intrinsic semimajor half-light radius.

The new response operator must distinguish pre-pixel PSF (convolve then
integrate once) from an effective PSF (response already included). Unknown
semantics block a real-JWST science claim but permit analytic mock development.
Legacy circular shift/block sum is a compatibility path, not that new operator.
Legacy fractions reference the supplied finite PSF sum, not an independently
measured infinite total flux. Record finite support loss, preparation loss,
stamp fraction and analytic Sérsic infinite flux separately.

Even native dimensions silently lose a row/column in the legacy downsampler.
Track this in `LEGACY-ISSUES.md`; keep the behavior in baseline tests. A future
new API must reject that legacy configuration by default until a validated
explicit origin/integration path supports it. M0 introduces no new renderer API.
