import asyncio
from main import ParallelProcessorWorker
from logger import setup_logger

class JobQueue:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.processing = {} # Track processing jobs
        self.lock = asyncio.Lock()
    
    async def add_job(self, job: dict) -> bool:
        async with self.lock:
            if job['job_id'] in self.processing:
                return False
            self.processing[job['job_id']] = job
            await self.queue.put(job)
            return True
    
    async def mark_complete(self, job_id: int):
        async with self.lock:
            self.processing.pop(job_id, None)

class GPUService:
    def __init__(self, num_concurrent_tasks: int = 4):
        self.logger = setup_logger('job_parallel', 'job_parallel.log')
        self.processor_worker = ParallelProcessorWorker(num_concurrent_tasks=num_concurrent_tasks)
        self.job_queue = JobQueue()
        self.processing_task = None
        self.running = False
    
    async def start(self):
        """Start the processing loop"""
        self.running = True
        self.processing_task = asyncio.create_task(self.process_queue())
        self.logger.info("Started GPU API Service")
    
    async def stop(self):
        """Stop the processing loop and cleanup"""
        self.logger.info("Stopping GPU API Service")
        self.running = False
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        # Wait for active tasks to complete
        if hasattr(self.processor_worker, 'active_tasks'):
            active_tasks = [task for task in self.processor_worker.active_tasks if not task.done()]
            if active_tasks:
                self.logger.info(f"Waiting for {len(active_tasks)} active tasks to complete")
                await asyncio.gather(*active_tasks, return_exceptions=True)
        self.logger.info("GPU API Service stopped")
    
    async def process_queue(self):
        """Main processing loop that handles queued jobs"""
        while True:
            try:
                try:
                    # Get next job from queue
                    # job = await self.job_queue.get()
                    job = await asyncio.wait_for(self.job_queue.queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                # Start processing in parallel
                asyncio.create_task(self.process_job(job))
            except Exception as e:
                self.logger.error(f"Error in process_queue: {str(e)}")
                await asyncio.sleep(1)
    
    async def process_job(self, job: dict):
        """Process a single job"""
        try:
            self.logger.info(f"Processing job: {job['job_id']}")
            await self.processor_worker.process_job(job)
        except Exception as e:
            self.logger.error(f"Error processing job {job['job_id']}: {str(e)}")
        finally:
            await self.job_queue.mark_complete(job['job_id'])

    def get_status(self) -> dict:
        """Get current processing status"""
        return {
            "queued_jobs": self.job_queue.queue.qsize(),
            "processing_jobs": len(self.job_queue.processing),
            "active_tasks": len(self.processor_worker.active_tasks),
            "max_concurrent": self.processor_worker.num_concurrent_tasks
        }
