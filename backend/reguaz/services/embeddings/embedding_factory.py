from backend.reguaz.services.embeddings.base import BaseEmbeddingService
from backend.reguaz.services.embeddings.e5 import E5EmbeddingService
from backend.reguaz.services.embeddings.bge_m3 import BGEM3EmbeddingService
from backend.reguaz.services.embeddings.jina_v3 import JinaV3EmbeddingService

class EmbeddingFactory:
    """
    Factory class to instantiate the appropriate embedding service.
    """
    
    @staticmethod
    def get_service(model_name: str) -> BaseEmbeddingService:
        """
        Get the embedding service for the specified model name.
        
        Args:
            model_name: The identifier of the model (e.g., 'e5', 'bge_m3', 'jina_v3')
            
        Returns:
            An instance of BaseEmbeddingService.
        """
        model_name_lower = model_name.lower().replace("-", "_")
        
        if model_name_lower == "e5":
            return E5EmbeddingService()
        elif model_name_lower == "bge_m3":
            return BGEM3EmbeddingService()
        elif model_name_lower == "jina_v3":
            return JinaV3EmbeddingService()
        else:
            raise ValueError(f"Unsupported embedding model: {model_name}")
