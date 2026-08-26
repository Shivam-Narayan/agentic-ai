"""Script to ingest PDFs from Google Drive and merge them into the vector index."""

import logging
from llama_index.readers.google import GoogleDriveReader
from .config import DRIVE_FOLDER_ID, setup_logging
from .rag import build_index

logger = logging.getLogger(__name__)

def main():
    setup_logging()
    
    if not DRIVE_FOLDER_ID:
        logger.error("GOOGLE_DRIVE_FOLDER_ID is not set in .env")
        return
        
    logger.info(f"Connecting to Google Drive folder: {DRIVE_FOLDER_ID}")
    logger.info("A browser window may open asking you to authenticate.")
    
    try:
        # The reader will look for credentials.json in the current working directory
        reader = GoogleDriveReader()
        
        drive_documents = reader.load_data(folder_id=DRIVE_FOLDER_ID)
        
        logger.info(f"Successfully downloaded {len(drive_documents)} documents from Google Drive.")
        
        logger.info("Merging Drive documents with local documents and rebuilding index...")
        build_index(extra_documents=drive_documents)
        logger.info("Ingestion complete. Vector store updated.")
        
    except Exception as e:
        logger.exception(f"Failed to ingest from Google Drive: {e}")
        logger.error("Make sure you have placed your OAuth credentials.json in the root directory.")

if __name__ == "__main__":
    main()
