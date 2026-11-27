"""Each registration selects an independent table schema and row iterator."""

from . import chime, libri, urgent, voicebank, wham
from .audio import layout_parser
from .vctk import vctk_parser


def directory_at(path, depth):
    directories = path.parts[:-1]
    return directories[depth] if len(directories) > depth else None


def directory_parser(name, depth=0):
    return layout_parser({name: "VARCHAR"}, lambda path: (directory_at(path, depth),))


def demand_fields(path):
    scene = path.parent.name
    classes = {
        "D": "domestic",
        "N": "nature",
        "O": "office",
        "P": "public",
        "S": "street",
        "T": "transport",
    }
    return scene, classes.get(scene[:1]), path.stem


demand = layout_parser(
    dict.fromkeys(("scene_code", "scene_class", "channel_label"), "VARCHAR"), demand_fields
)
musan = layout_parser(
    {"category": "VARCHAR", "source": "VARCHAR"},
    lambda path: (directory_at(path, 0), directory_at(path, 1)),
)

# Keys are stable SQL/link names. Directory basenames may vary between machines.
CATALOG = {
    "chime_3": ("CORPUS_CHIME3", chime.parser),
    "demand": ("CORPUS_DEMAND", demand),
    "demand_48khz": ("CORPUS_DEMAND48", demand),
    "babble": ("CORPUS_BABBLE", layout_parser()),
    "ssn": ("CORPUS_SSN", layout_parser()),
    "dns_challenge_5": ("CORPUS_DNS5", directory_parser("collection", 1)),
    "dns_challenge_2020": ("CORPUS_DNS2020", directory_parser("collection", 1)),
    "ms_snsd": ("CORPUS_MS_SNSD", directory_parser("collection")),
    "librispeech": ("CORPUS_LIBRISPEECH", libri.librispeech),
    "libritts": ("CORPUS_LIBRITTS", libri.libritts),
    "openslr_26": ("CORPUS_OPENSLR26", directory_parser("collection")),
    "openslr_28": ("CORPUS_OPENSLR28", directory_parser("collection", 1)),
    "urgent_2024": ("CORPUS_URGENT2024", urgent.parser),
    "vctk_0_80": ("CORPUS_VCTK", vctk_parser("wav48")),
    "vctk_0_92": ("CORPUS_VCTK_0_92", vctk_parser("wav48_silence_trimmed")),
    "vctk_rva": ("CORPUS_VCTK_RVA", vctk_parser("wav48_silence_trimmed")),
    "voicebank_demand": ("CORPUS_VOICEBANK_DEMAND", voicebank.parser),
    "wham_48khz": ("CORPUS_WHAM48", wham.parser),
    "musan": ("CORPUS_MUSAN", musan),
}
