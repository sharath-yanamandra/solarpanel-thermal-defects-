import torch
from ultralytics import YOLO
from typing import Dict, Optional, List
import threading
import asyncio

from logger import setup_logger
from config import Config

class ModelInstance:
    def __init__(self, model: YOLO, model_type: str):
        self.model = model
        self.model_type = model_type
        self.in_use = False
        self.lock = threading.Lock()
        # track memory usage for this instance
        # self.initial_memory = torch.cuda.memory_allocated()

class SingleGPUModelManager:
    def __init__(self, instances_per_type: int = 50, memory_threshold: float = 0.85):
        """
        Initialize model manager for single GPU
        Args:
            instances_per_type: Number of instances per model type to load
        """
        self.logger = setup_logger('gpu_model_manager', 'gpu_model_manager.log')
        self.logger.info("Initializing single_gpu_model_manager.")

        self.instances_per_type = instances_per_type
        self.memory_threshold = memory_threshold
        self.total_gpu_memory = torch.cuda.get_device_properties(0).total_memory

        self.model_instances: Dict[str, List[ModelInstance]] = {
            'hotspot': []
        }
        self.model_locks = {
            'hotspot': threading.Lock()
        }
        self.initialize_models()
    
    def _check_memory_usage(self) -> float:
        """
        Check current GPU memory usage
        Returns:
            float: Fraction of GPU memory currently in use
        """
        allocated = torch.cuda.memory_allocated()
        return allocated / self.total_gpu_memory
    
    def initialize_models(self):
        """Initialize multiple instances of each model type on single GPU"""
        try:
            # Ensure CUDA is available
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")
            
            self.logger.info(f"Initializing {self.instances_per_type} instances per model type on GPU")

            # Initialize models with CUDA memory management
            with torch.cuda.device(0):
                # Initialize hotspot models
                for i in range(self.instances_per_type):
                    hotspot_model = YOLO(Config.DETECTION_MODEL_PATH)
                    hotspot_model.to('cuda')
                    self.model_instances['hotspot'].append(
                        ModelInstance(hotspot_model, 'hotspot')
                    )
                    self.logger.info(f"Detection Model: {i} initialized!")

            # Log GPU memory usage
            allocated = torch.cuda.memory_allocated(0)
            reserved = torch.cuda.memory_allocated(0)
            self.logger.info(f"GPU Memory: Allocated={allocated/1e9:.2f}GB, Reserved={reserved/1e9:.2f}GB")
        
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {str(e)}", exc_info=True)
            raise
    
    async def get_available_model(self, model_type: str) -> Optional[ModelInstance]:
        """Get an available model instance of the specified type"""
        while True:
            with self.model_locks[model_type]:
                for instance in self.model_instances[model_type]:
                    self.logger.info(f"Checking instance: {instance.model_type}")
                    with instance.lock:
                        if not instance.in_use:
                            instance.in_use = True
                            self.logger.info(f"Returning instance: {instance.model_type}")
                            return instance
            # If no instance is available, wait briefly before checking again
            await asyncio.sleep(1)
    
    async def release_model(self, instance: ModelInstance):
        """Release a model instance back to the pool"""
        with instance.lock:
            instance.in_use = False
            # Clear CUDA cache for this model if needed
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()
        self.logger.info(f"Released {instance.model_type} model instance")
    
    # def _clear_instance_memory(self, instance: ModelInstance):
    #     """
    #     Clear memory specifically associated with this model instance
    #     """
    #     try:
    #         # Clear stored intermediate results
    #         if hasattr(instance, 'intermediate_results'):
    #             instance.intermediate_results.clear()
            
    #         # Clear last computation if it exists
    #         if hasattr(instance, 'last_computation'):
    #             del instance.last_computation
                
    #         # Clear model gradients
    #         if hasattr(instance.model, 'zero_grad'):
    #             instance.model.zero_grad(set_to_none=True)  # More efficient than just zero_grad()
                
    #         # Remove any references to stored tensors
    #         for attr_name in dir(instance):
    #             attr = getattr(instance, attr_name)
    #             if isinstance(attr, torch.Tensor):
    #                 delattr(instance, attr_name)
                    
    #         # Force garbage collection for this instance
    #         gc.collect()
    #         if torch.cuda.is_available():
    #             torch.cuda.synchronize()  # Ensure CUDA operations are complete
                
    #     except Exception as e:
    #         self.logger.error(f"Error clearing instance memory: {str(e)}", exc_info=True)
    
    # async def release_model(self, instance: ModelInstance):
    #     """
    #     Release a model instance back to the pool with targeted memory cleanup
    #     """
    #     try:
    #         with instance.lock:
    #             # Clear instance-specific memory
    #             self._clear_instance_memory(instance)
                
    #             # Check overall memory usage
    #             current_usage = self._check_memory_usage()
    #             self.logger.debug(f"Current GPU memory usage: {current_usage:.2%}")
                
    #             # If memory usage is above threshold, perform additional cleanup
    #             if current_usage > self.memory_threshold:
    #                 self.logger.warning(
    #                     f"High memory usage detected ({current_usage:.2%}). "
    #                     "Performing additional cleanup."
    #                 )
    #                 torch.cuda.empty_cache()
                    
    #                 # Log memory after cleanup
    #                 new_usage = self._check_memory_usage()
    #                 self.logger.info(
    #                     f"Memory usage after cleanup: {new_usage:.2%} "
    #                     f"(freed {(current_usage - new_usage) * 100:.2f}%)"
    #                 )
                
    #             # Finally mark the instance as not in use
    #             instance.in_use = False
                
    #     except Exception as e:
    #         self.logger.error(f"Error in release_model: {str(e)}", exc_info=True)
    #         raise
