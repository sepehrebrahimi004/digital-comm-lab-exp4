"""Digital Communications Lab, Experiment 4: BPSK, QPSK and 16-QAM.

Reconstruction in NumPy of the three GNU Radio Companion flowgraphs used in
the lab (screenshots in flowgraphs/):

  Random Source -> Chunks to Symbols -> (+ Noise Source) -> Constellation Sink
  16-QAM additionally: Rational Resampler (x10) -> x 100 kHz carrier -> sinks

and extended with symbol/bit-error measurements against theory.

GNU Radio's complex Gaussian Noise Source with amplitude A has total power
A^2 (A^2/2 per I/Q component); that convention is used throughout.

    python digital_modulation_grc.py      -> results/*.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402
from scipy.special import erfc  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
rng = np.random.default_rng(1405)

TABLES = {
    "BPSK": np.array([-1, 1], dtype=complex),
    "QPSK (lab table)": np.array([-1, 1j, 1, -1j]),          # natural order, as in the lab
    "QPSK (Gray)": np.array([-1, 1j, -1j, 1]),                # neighbours differ in one bit
    "16-QAM": np.array([x + 1j * y for y in (3, 1, -1, -3) for x in (-3, -1, 1, 3)]),
}


def qfunc(x):
    return 0.5 * erfc(x / np.sqrt(2))


def noise(n, amplitude):
    return amplitude / np.sqrt(2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))


def detect(rx, table):
    return np.argmin(np.abs(rx[:, None] - table[None, :]), axis=1)


def bit_errors(a, b, k):
    x = np.bitwise_xor(a, b)
    return sum(((x >> i) & 1).sum() for i in range(k))


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=130)
    plt.close(fig)


def constellations():
    fig, axs = plt.subplots(2, 3, figsize=(13, 8.4))
    for row, amp in enumerate((0.2, 0.6)):
        for col, name in enumerate(("BPSK", "QPSK (lab table)", "16-QAM")):
            table = TABLES[name]
            sym = rng.integers(0, len(table), 1024)
            rx = table[sym] + noise(len(sym), amp * (3 if name == "16-QAM" else 1))
            ax = axs[row, col]
            ax.scatter(rx.real, rx.imag, s=4, alpha=0.5)
            ax.scatter(table.real, table.imag, s=40, c="r")
            for i, p in enumerate(table):
                ax.annotate(format(i, f"0{int(np.log2(len(table)))}b"), (p.real, p.imag),
                            textcoords="offset points", xytext=(4, 4), fontsize=7, color="darkred")
            lim = 1.3 * np.max(np.abs(table.real)) + 1
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            ax.grid(alpha=0.3)
            na = amp * (3 if name == "16-QAM" else 1)
            ax.set_title(f"{name}, noise amplitude {na:.1f}")
    save(fig, "constellations.png")


def error_rates():
    amps = np.linspace(0.05, 1.5, 30)
    n = 200_000
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    for name, table in TABLES.items():
        k = int(np.log2(len(table)))
        es = np.mean(np.abs(table) ** 2)
        ser, ber = [], []
        for a in amps:
            sym = rng.integers(0, len(table), n)
            det = detect(table[sym] + noise(n, a), table)
            ser.append(np.mean(det != sym))
            ber.append(bit_errors(sym, det, k) / (n * k))
        esn0_db = 10 * np.log10(es / amps ** 2)
        h = axs[0].semilogy(esn0_db, np.maximum(ser, 1e-6), "o", ms=4, label=name)[0]
        axs[1].semilogy(esn0_db, np.maximum(ber, 1e-6), "o", ms=4, color=h.get_color(), label=name)
        g = es / amps ** 2                                       # Es/N0 (linear), N0 = A^2
        if name == "BPSK":
            th = qfunc(np.sqrt(2 * g))
        elif name.startswith("QPSK"):
            p = qfunc(np.sqrt(g))
            th = 1 - (1 - p) ** 2
        else:
            p = 1.5 * qfunc(np.sqrt(g / 5))
            th = 1 - (1 - p) ** 2
        if name != "QPSK (Gray)":
            axs[0].semilogy(esn0_db, th, "-", color=h.get_color(), lw=1)
    axs[0].set_title("Symbol error rate (markers: simulation, lines: theory)")
    axs[1].set_title("Bit error rate: the lab's QPSK table is not Gray-coded")
    for ax in axs:
        ax.set_xlabel("E_s/N_0 (dB)  [= mean symbol energy / A²]")
        ax.set_ylim(1e-5, 1)
        ax.set_xlim(-5, 22)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    save(fig, "error_rates.png")


def qam_upconversion(noise_amp=0.0):
    """16-QAM at 32 kS/s -> Rational Resampler (10/1) -> x exp(j 2pi 100k t) at 320 kS/s."""
    fs1, fs2, fc = 32e3, 320e3, 100e3
    table = TABLES["16-QAM"]
    sym = rng.integers(0, 16, 4000)
    base = table[sym] + noise(len(sym), noise_amp)            # one sample per symbol
    up = resample_poly(base, 10, 1)                           # windowed-sinc interpolation
    t = np.arange(len(up)) / fs2
    ifsig = up * np.exp(2j * np.pi * fc * t)
    down = ifsig * np.exp(-2j * np.pi * fc * t)               # added: coherent down-conversion
    rec = down[::10]
    return fs2, fc, base, up, ifsig, rec


def qam_plots():
    fs2, fc, base, up, ifsig, rec = qam_upconversion()
    nfft = 2048
    win = np.hanning(nfft)
    segs = [ifsig[i:i + nfft] * win for i in range(0, len(ifsig) - nfft, nfft // 2)]
    p = np.mean([np.abs(np.fft.fftshift(np.fft.fft(s))) ** 2 for s in segs], axis=0)
    f = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / fs2))
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    axs[0].plot(f / 1e3, 10 * np.log10(p / p.max()))
    axs[0].axvspan((fc - 16e3) / 1e3, (fc + 16e3) / 1e3, color="g", alpha=0.1, label="100 kHz ± R_s/2")
    axs[0].set_ylim(-80, 3)
    axs[0].set_xlabel("frequency (kHz)")
    axs[0].set_ylabel("dB")
    axs[0].set_title("After the ×10 resampler and the 100 kHz mixer")
    axs[0].legend(fontsize=8)
    axs[1].scatter(ifsig.real[:3000], ifsig.imag[:3000], s=2, alpha=0.4)
    axs[1].set_aspect("equal")
    axs[1].set_title("Constellation sink after the mixer:\nthe points rotate at 100 kHz")
    axs[2].scatter(rec.real, rec.imag, s=4, alpha=0.5)
    axs[2].set_aspect("equal")
    axs[2].set_title("After coherent down-conversion and\nsampling every 10th sample (added)")
    for ax in axs:
        ax.grid(alpha=0.3)
    save(fig, "qam16_upconversion.png")
    err = np.max(np.abs(rec[50:-50] - base[50:-50]))
    print(f"16-QAM: worst-case reconstruction error after up/down-conversion = {err:.2e}")


if __name__ == "__main__":
    constellations()
    error_rates()
    qam_plots()
