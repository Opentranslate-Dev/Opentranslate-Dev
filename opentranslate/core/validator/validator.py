from typing import Dict, List, Optional
from ..blockchain.contracts import BlockchainContracts
from ...ai.models.translation_model import TranslationModel

class Validator:
    def __init__(
        self,
        blockchain: BlockchainContracts,
        ai_model: TranslationModel,
        account: str
    ):
        self.blockchain = blockchain
        self.ai_model = ai_model
        self.account = account

    async def validate_translation(
        self,
        translation_id: str,
        score: int,
        feedback: str
    ) -> bool:
        # Get translation details
        translation = await self.blockchain.get_translation_status(translation_id)
        
        # Get AI quality score for comparison
        ai_score = self.ai_model.get_quality_score(
            translation["source_text"],
            translation["target_text"]
        )
        
        # Submit validation
        success = await self.blockchain.validate_translation(
            self.account,
            translation_id,
            score,
            feedback
        )
        
        return success

    async def get_pending_validations(self) -> List[Dict]:
        # Get list of translations pending validation
        pass

    async def get_rewards(self) -> Dict:
        return await self.blockchain.get_rewards(self.account) 