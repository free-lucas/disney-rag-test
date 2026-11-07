from __future__ import annotations

from dataclasses import dataclass, field

from ingest.extract import ImageAsset


@dataclass
class ImageFilterConfig:
    min_width: int = 180
    min_height: int = 180
    min_area: int = 32400
    max_per_document: int = 80


@dataclass
class FilteredImage:
    image: ImageAsset
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            'source_file': self.image.source_file,
            'image_path': self.image.path,
            'image_hash': self.image.image_hash,
            'page': self.image.page,
            'slide': self.image.slide,
            'width': self.image.width,
            'height': self.image.height,
            'reason': self.reason,
        }


@dataclass
class ImageFilterResult:
    kept: list[ImageAsset] = field(default_factory=list)
    filtered: list[FilteredImage] = field(default_factory=list)


def filter_images(
    images: list[ImageAsset],
    *,
    config: ImageFilterConfig | None = None,
    seen_hashes: set[str] | None = None,
) -> ImageFilterResult:
    configuration = config or ImageFilterConfig()
    hashes = seen_hashes if seen_hashes is not None else set()
    result = ImageFilterResult()
    for image in images:
        if image.image_hash in hashes:
            result.filtered.append(FilteredImage(image, 'duplicate'))
            continue
        if image.width is not None and image.width < configuration.min_width:
            result.filtered.append(FilteredImage(image, 'too_narrow'))
            continue
        if image.height is not None and image.height < configuration.min_height:
            result.filtered.append(FilteredImage(image, 'too_short'))
            continue
        if image.width is not None and image.height is not None and image.width * image.height < configuration.min_area:
            result.filtered.append(FilteredImage(image, 'too_small'))
            continue
        if len(result.kept) >= configuration.max_per_document:
            result.filtered.append(FilteredImage(image, 'over_document_limit'))
            continue
        hashes.add(image.image_hash)
        result.kept.append(image)
    return result


def apply_captions(images: list[ImageAsset], captions: dict[str, str]) -> None:
    for image in images:
        if image.path in captions:
            image.caption = captions[image.path]
