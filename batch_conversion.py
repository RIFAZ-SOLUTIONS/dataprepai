import json
import logging
import time
from collections.abc import Iterable
from pathlib import Path

import yaml
from docling_core.types.doc import ImageRefMode

from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

_log = logging.getLogger(__name__)

USE_V2 = True
USE_LEGACY = False
BATCH_SIZE = 20  # Process files in batches


def export_documents(
    conv_results: Iterable[ConversionResult],
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    failure_count = 0
    partial_success_count = 0

    for conv_res in conv_results:
        if conv_res.status == ConversionStatus.SUCCESS:
            success_count += 1
            doc_filename = conv_res.input.file.stem

            if USE_V2:
                # Export Docling document format to markdown:
                with (output_dir / f"{doc_filename}.md").open("w") as fp:
                    fp.write(conv_res.document.export_to_markdown())

            if USE_LEGACY:
                # Export Markdown format:
                with (output_dir / f"{doc_filename}.legacy.md").open(
                    "w", encoding="utf-8"
                ) as fp:
                    fp.write(conv_res.legacy_document.export_to_markdown())

        elif conv_res.status == ConversionStatus.PARTIAL_SUCCESS:
            _log.info(
                f"Document {conv_res.input.file} was partially converted with the following errors:"
            )
            for item in conv_res.errors:
                _log.info(f"\t{item.error_message}")
            partial_success_count += 1
        else:
            _log.info(f"Document {conv_res.input.file} failed to convert.")
            failure_count += 1

    _log.info(
        f"Processed {success_count + partial_success_count + failure_count} docs, "
        f"of which {failure_count} failed "
        f"and {partial_success_count} were partially converted."
    )
    return success_count, partial_success_count, failure_count


def process_in_batches(doc_converter, files_to_convert, output_folder, batch_size=BATCH_SIZE):
    """Process files in batches of the specified size."""
    total_success = 0
    total_partial = 0
    total_failure = 0
    total_batches = (len(files_to_convert) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(files_to_convert))
        batch_files = files_to_convert[start_idx:end_idx]
        
        _log.info(f"Processing batch {batch_num + 1}/{total_batches} with {len(batch_files)} files")
        
        batch_start_time = time.time()
        conv_results = doc_converter.convert_all(
            batch_files,
            raises_on_error=False,
        )
        success, partial, failure = export_documents(
            conv_results, output_dir=output_folder
        )
        
        batch_end_time = time.time() - batch_start_time
        _log.info(f"Batch {batch_num + 1}/{total_batches} completed in {batch_end_time:.2f} seconds")
        
        total_success += success
        total_partial += partial
        total_failure += failure
    
    return total_success, total_partial, total_failure


def main():
    logging.basicConfig(level=logging.INFO)

    input_folder = Path("./taxlawsanddata-server")
    output_folder = Path("./data_md")
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Get all PDF files from input folder
    input_doc_paths = list(input_folder.glob("**/*.pdf"))
    _log.info(f"Found {len(input_doc_paths)} PDF files in input folder")
    
    # Filter out files that have already been converted
    files_to_convert = []
    for pdf_path in input_doc_paths:
        md_path = output_folder / f"{pdf_path.stem}.md"
        if md_path.exists():
            _log.info(f"Skipping {pdf_path.name} as it has already been converted")
        else:
            files_to_convert.append(pdf_path)
    
    _log.info(f"{len(files_to_convert)} files need conversion")
    
    if not files_to_convert:
        _log.info("No files to convert. Exiting.")
        return

    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_page_images = False

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options, backend=DoclingParseV4DocumentBackend
            )
        }
    )

    start_time = time.time()

    # Process files in batches of BATCH_SIZE
    success_count, partial_success_count, failure_count = process_in_batches(
        doc_converter,
        files_to_convert,
        output_folder,
        BATCH_SIZE
    )

    end_time = time.time() - start_time

    _log.info(f"Document conversion complete in {end_time:.2f} seconds.")
    _log.info(f"Total results: {success_count} successes, {partial_success_count} partial successes, {failure_count} failures")

    if failure_count > 0:
        _log.warning(
            f"The conversion failed for {failure_count} out of {len(files_to_convert)} files."
        )


if __name__ == "__main__":
    main()
