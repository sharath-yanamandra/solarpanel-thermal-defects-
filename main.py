import asyncio
from typing import List

from logger import setup_logger
from model_manager import SingleGPUModelManager
from singlegpu_videoprocessor import ParallelVideoProcessor

class ParallelProcessorWorker:
    def __init__(self, num_concurrent_tasks: int = 4):
        self.num_concurrent_tasks = num_concurrent_tasks
        self.running = False
        self.logger = setup_logger('main', 'main.log')
        self.logger.info("Initializing main worker.")

        # Initialize GPU model manager with instances per type
        self.model_manager = SingleGPUModelManager(instances_per_type=num_concurrent_tasks)
        self.processor = ParallelVideoProcessor(self.model_manager)

        # Track active tasks
        self.active_tasks: List[asyncio.Task] = []
        self.semaphore = asyncio.Semaphore(num_concurrent_tasks)

    async def process_job(self, job: dict):
        """Process a single job with semaphore control and user verification"""
        async with self.semaphore: # Control concurrent processing
            try:
                video_path = job.get('video_path')
                video_type = job.get('video_type')
                try:
                    # Process video
                    success = await self.processor.process_video(
                        video_path,
                        video_type
                    )
                    self.logger.info(f"processing completed successfully: {success}")

                except Exception as e:
                    self.logger.error(f"Error processing job: {str(e)}", exc_info=True)

            except Exception as e:
                self.logger.error(f"Error in process_job: {str(e)}", exc_info=True)

if __name__ == "__main__":
    processor_worker = ParallelProcessorWorker(num_concurrent_tasks=4)
    asyncio.run(processor_worker.process_job({'video_path': 'C:/Users/DELL/Desktop/ctrlS nagpur/gpu-debug-main/test.MP4', 'video_type': 'hotspot'}))