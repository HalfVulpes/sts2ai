# Ownership and dependencies

Original project code is licensed under **GPL-2.0-only** in [LICENSE](LICENSE),
as chosen by Xiangqin Chen. This is the kernel's GPL version choice; this project
does not adopt the Linux syscall exception or imply affiliation with Linux.
See the [Linux licensing documentation](https://docs.kernel.org/process/license-rules.html).

Slay the Spire 2, its assemblies, game rules implementation, artwork, localization,
and other assets remain the property of their respective rights holders. They are
not licensed by this repository and are not part of its source distribution.
The importer reads an installation supplied by the user; it does not download the
game. This project is independent research and is not endorsed by Mega Crit.

The headless host references locally supplied game assemblies, including GodotSharp,
Harmony, MonoMod, SmartFormat, Sentry, Steamworks.NET, JetBrains.Annotations and
System.IO.Hashing. Their own license notices govern them. Building against those
files does not make them GPL project code. The release source archive excludes all
DLLs, PCKs and compiled mod distributions. Review the upstream notices and applicable
distribution terms before distributing any separately built binary bundle.

Python training uses PyTorch, NumPy and TensorBoard; development uses pytest.
The installed packages retain their licenses. Native numerical inference may use
the host's BLAS libraries. No third-party library source is vendored in this release.

External baseline software is downloaded separately outside this source tree:

- [AlphaSpire](https://github.com/AlphaSpire2/alphaspire), AGPL-3.0-or-later source,
  with a separately packaged simulator. Its binaries and model files are not
  redistributed or incorporated into this repository.
- [Spire Mentor](https://github.com/tfriedel/spire-mentor), MIT source. The downloaded
  snapshot remains external. We do not relabel its MCTS-assisted result as zero-search.

This repository contains source code and short documentation only. The trained parameters and the
in-game mod package are published as GitHub release assets under the project's GPL-2.0-only licence;
they contain learned weights and our own compiled code only, no game assets or vocabulary. Evaluation
records, protocols and logs are published separately with the paper; the language-model logs omit the
rendered observations, which contain the game's own text.
