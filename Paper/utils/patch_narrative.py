import re
import sys

file_path = 'c:/Users/pathi/OneDrive/Desktop/LAST/Paper/main.tex'
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# 1. ABSTRACT REPLACEMENT
abstract_old = r"""These free structural priors reduce the burden on downstream learnable
spatial modules, enabling a compact \emph{EfficientZero Block} pairing a
$K{=}3$ partition graph convolution with a depthwise-separable temporal
convolution (DS-TCN).
Three models cover the edge-to-server spectrum:
\textbf{Nano} (97\,K params) with single-stream early fusion and Temporal
Landmark Attention;
\textbf{Small} (267\,K) with two-backbone late fusion and cross-stream
gating; and
\textbf{Large} (1.67\,M) with three-backbone mid-network fusion matching
the EfficientGCN-B4 paradigm.
On NTU\,RGB+D\,60 cross-subject the trio achieves
85.6\,\%/87.6\,\%/92.5\,\% without distillation and
88.5\,\%/90.2\,\%/92.5\,\% with knowledge distillation from the large
teacher, delivering up to $10{\times}$ fewer parameters than comparable
baselines."""

abstract_new = r"""These free structural priors reduce the burden on downstream learnable
modules. By intelligently positioning Joint Embedding and spatial attention
\emph{before} the $K{=}3$ graph convolution, the compact \emph{EfficientZero Block}
operates on an already anatomically-refined manifold.
Three models cover the edge-to-server spectrum:
\textbf{Nano} (97\,K) and \textbf{Small} (250\,K) act as hyper-efficient,
quantisation-friendly solutions demanding ultra-low FLOPs, while \textbf{Large} (1.67\,M)
targets maximum capacity.
On NTU\,RGB+D\,60 cross-subject the trio achieves
85.6\,\%/87.6\,\%/92.5\,\% without distillation and
88.5\,\%/89.8\,\%/92.5\,\% with knowledge distillation.
Strikingly, our Small variant efficiently beats EfficientGCN-B0 with fewer parameters,
delivering up to $10{\times}$ parameter reduction over comparable baselines
and enabling faster edge inference."""

if abstract_old not in text:
    print("WARNING: Abstract string not found!")
else:
    text = text.replace(abstract_old, abstract_new)


# 2. INTRODUCTION REPLACEMENT
intro_old = r"""Our answer is \textbf{ShiftFuse-Zero}, a family of three models that
achieves a new Pareto-optimal trade-off across the accuracy--parameter
spectrum.
SFZ-Nano reaches \textbf{88.5\%} on NTU-60 X-Sub at only 97\,K
parameters --- outperforming the 6.9\,M-parameter AGCN while fitting
in under 400\,KB of memory.
SFZ-Large achieves \textbf{92.5\%}, surpassing EfficientGCN-B4 at
a competitive parameter budget (1.67\,M).
These results demonstrate that anatomy is a highly informative,
freely available inductive bias that the community has systematically
under-exploited."""

intro_new = r"""Our answer is \textbf{ShiftFuse-Zero}, a family of three models that
achieves a new Pareto-optimal trade-off across the accuracy--parameter
spectrum.
While our primary innovation lies in the zero-parameter anatomical priors of
BRASP and SGPShift, these modules inherently catalyze hyper-efficient
downstream architectures.
SFZ-Nano (97\,K) and SFZ-Small (250\,K) redefine the edge-deployment frontier
by yielding ultra-low FLOPs, less learnable parameters, and quantisation-friendly
operations for faster inference. On NTU-60 X-Sub, SFZ-Nano reaches \textbf{88.5\%}
--- outperforming the 6.9\,M-parameter AGCN while fitting in under 400\,KB.
Strikingly, our Small-Late variant achieves 89.8\%, strictly beating EfficientGCN-B0
with fewer parameters.
SFZ-Large limits the high-capacity spectrum, pushing the absolute accuracy
frontier to \textbf{92.5\%} at 1.67\,M parameters.
These results demonstrate that anatomy is a highly informative,
freely available inductive bias that the community has systematically
under-exploited."""

if intro_old not in text:
    print("WARNING: Intro string not found!")
else:
    text = text.replace(intro_old, intro_new)


# 3. BLOCK ORDERING REPLACEMENT
block_old = r"""\item \textbf{Residual} $\oplus$: skip from block input with $1{\times}1$
  projection when shape changes.
\end{enumerate}

\paragraph{Temporal Landmark Attention (TLA).}"""

block_new = r"""\item \textbf{Residual} $\oplus$: skip from block input with $1{\times}1$
  projection when shape changes.
\end{enumerate}

\paragraph{Intelligent Block Ordering.}
The defining architectural insight of the EfficientZero Block is placing the
Joint Embedding and STC-Attention \emph{before} the Graph Convolution.
While BRASP and SGPShift establish the zero-parameter anatomical foundation,
this subsequent lightweight bottleneck maps the raw topological variables
into a semantically rich, joint-specific latent space. This strategic placement
liberates the downstream GCN from fundamental feature-extraction duties,
allowing it to focus entirely on efficient, localized message passing.
Consequently, we achieve superior representation power and much faster inference
speeds with drastically fewer learnable parameters.

\paragraph{Temporal Landmark Attention (TLA).}"""

if block_old not in text:
    print("WARNING: Block string not found!")
else:
    text = text.replace(block_old, block_new)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(text)

print("Updates successfully applied to main.tex.")
