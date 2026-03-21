import re

file_path = 'c:/Users/pathi/OneDrive/Desktop/LAST/Paper/main.tex'
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

abstract_old = r"""These free structural priors reduce the burden on downstream learnable
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

abstract_new = r"""These free structural priors reduce the burden on downstream learnable
modules. By intelligently placing Joint Embedding and spatial attention
\emph{before} the graph convolution, the \emph{EfficientZero Block}
operates on an already anatomically-refined manifold.
Three models cover the edge-to-server spectrum: \textbf{Nano} (97\,K) and
\textbf{Small} (250\,K) are hyper-efficient, quantisation-friendly solutions
with ultra-low FLOPs, while \textbf{Large} (1.67\,M) targets maximum capacity.
On NTU\,RGB+D\,60 cross-subject the trio achieves 85.6\,\%/87.6\,\%/92.5\,\%
without distillation and 88.5\,\%/89.8\,\%/92.5\,\% with distillation.
Strikingly, our Small variant strictly beats EfficientGCN-B0 with fewer parameters,
delivering up to $10{\times}$ parameter reduction over comparable baselines
for fast edge inference."""

text = text.replace(abstract_old, abstract_new)

intro_old = r"""While our primary innovation lies in the zero-parameter anatomical priors of
BRASP and SGPShift, these modules inherently catalyze hyper-efficient
downstream architectures.
SFZ-Nano (97\,K) and SFZ-Small (250\,K) redefine the edge-deployment frontier
by yielding ultra-low FLOPs, less learnable parameters, and quantisation-friendly
operations for faster inference. On NTU-60 X-Sub, SFZ-Nano reaches \textbf{88.5\%}
--- outperforming the 6.9\,M-parameter AGCN while fitting in under 400\,KB.
Strikingly, our Small-Late variant achieves 89.8\%, strictly beating EfficientGCN-B0
with fewer parameters.
SFZ-Large limits the high-capacity spectrum, pushing the absolute accuracy
frontier to \textbf{92.5\%} at 1.67\,M parameters."""

intro_new = r"""While our core contribution remains the zero-parameter priors of BRASP and SGPShift,
these modules catalyze hyper-efficient downstream architectures. SFZ-Nano (97\,K) and
SFZ-Small (250\,K) redefine edge-deployment by yielding ultra-low FLOPs and
quantisation-friendly operations for fast inference. On NTU-60 X-Sub, SFZ-Nano reaches
\textbf{88.5\%} --- outperforming the 6.9\,M AGCN within 400\,KB. Strikingly,
our Small variant achieves 89.8\%, strictly beating EfficientGCN-B0 with fewer parameters.
SFZ-Large limits the high-capacity spectrum, pushing the absolute accuracy
frontier to \textbf{92.5\%} at 1.67\,M parameters."""

text = text.replace(intro_old, intro_new)

block_old = r"""\paragraph{Intelligent Block Ordering.}
The defining architectural insight of the EfficientZero Block is placing the
Joint Embedding and STC-Attention \emph{before} the Graph Convolution.
While BRASP and SGPShift establish the zero-parameter anatomical foundation,
this subsequent lightweight bottleneck maps the raw topological variables
into a semantically rich, joint-specific latent space. This strategic placement
liberates the downstream GCN from fundamental feature-extraction duties,
allowing it to focus entirely on efficient, localized message passing.
Consequently, we achieve superior representation power and much faster inference
speeds with drastically fewer learnable parameters."""

block_new = r"""\paragraph{Intelligent Block Ordering.}
We deliberately place Joint Embedding and STC-Attention \emph{before} the Graph
Convolution. Following the zero-parameter foundation of BRASP and SGPShift, this
lightweight bottleneck maps raw topology into a rich, joint-specific latent space.
This liberates the downstream GCN to focus entirely on efficient, localized message
passing, achieving superior representation power and fast inference with drastically
fewer parameters."""

text = text.replace(block_old, block_new)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(text)

fig4_path = 'c:/Users/pathi/OneDrive/Desktop/LAST/Paper/images/fig4_block.tex'
with open(fig4_path, 'r', encoding='utf-8') as f:
    fig4 = f.read()

fig4 = fig4.replace('below=0.3cm', 'below=0.15cm')
fig4 = fig4.replace('right=0.15cm', 'right=0.1cm')

with open(fig4_path, 'w', encoding='utf-8') as f:
    f.write(fig4)

print("Trim script executed.")
