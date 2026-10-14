# Experiment summary

This summary preserves the headline figures stated in the supplied research paper. It is a transcription, not an independent rerun of the experiments.

## Trained residual U-Net

| Epoch | Training L1 | Validation L1 | Validation PSNR |
|---:|---:|---:|---:|
| 1 | 0.1232 | 0.1277 | 16.38 dB |
| 10 | 0.1016 | 0.1152 | 17.30 dB |
| 20 | 0.0922 | 0.0880 | 19.17 dB |
| 30 | 0.0896 | 0.0858 | 19.10 dB |
| 40 | 0.0863 | 0.0978 | 18.40 dB |
| 50 | 0.0821 | 0.0842 | 19.61 dB |
| 62 | 0.0805 | 0.0803 | 20.15 dB |
| **74** | **0.0784** | **0.0732** | **20.63 dB** |
| 80 | 0.0750 | 0.0871 | 19.64 dB |
| 90 | 0.0767 | 0.0951 | 18.75 dB |
| 100 | 0.0744 | 0.0917 | 18.95 dB |

## Classical baselines

| Method | Reported composite score | Brightness | Gradient consistency | Entropy | Halo penalty |
|---|---:|---:|---:|---:|---:|
| CLAHE | 0.874 | 112.3 | 0.965 | 7.21 | 0.12 |
| Exposure Fusion | 0.872 | 108.7 | 0.961 | 7.18 | 0.14 |
| Gamma | 0.801 | 134.2 | 0.912 | 6.98 | 0.23 |
| MSRCR | 0.763 | 118.5 | 0.878 | 7.05 | 0.31 |
| MSR | 0.721 | 105.1 | 0.843 | 6.87 | 0.28 |
| SSR | 0.698 | 102.3 | 0.821 | 6.74 | 0.29 |
| Zero-DCE-inspired curve | 0.651 | 127.8 | 0.486 | 7.31 | 0.42 |
| Histogram Equalization | 0.589 | 141.2 | 0.712 | 7.44 | 0.51 |

The paper describes a small validation split and limited testing beyond the LOL dataset's indoor-image distribution. Scores should be interpreted within that experimental setup, not as universal rankings. The classical implementations in this repository are organized for practical use; exact numerical reproduction of the paper's composite scores requires the original scoring implementation, normalization constants, and evaluation images.
