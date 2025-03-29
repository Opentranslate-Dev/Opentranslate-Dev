from typing import Dict, List, Optional
from ..blockchain.contracts import BlockchainContracts
from ...ai.models.translation_model import TranslationModel

class Translator:
    def __init__(
        self,
        blockchain: BlockchainContracts,
        ai_model: TranslationModel,
        account: str
    ):
        self.blockchain = blockchain
        self.ai_model = ai_model
        self.account = account

    async def submit_translation(
        self,
        source_text: str,
        target_text: str,
        source_lang: str,
        target_lang: str,
        domain: str
    ) -> str:
        # Get AI suggestions
        ai_translation = self.ai_model.translate(source_text, source_lang, target_lang)
        quality_score = self.ai_model.get_quality_score(source_text, target_text)
        domain_suggestions = self.ai_model.get_domain_suggestions(source_text)

        # Submit to blockchain
        translation_id = await self.blockchain.submit_translation(
            self.account,
            source_text,
            target_text,
            source_lang,
            target_lang,
            domain
        )

        return translation_id

    async def get_translation_status(self, translation_id: str) -> Dict:
        return await self.blockchain.get_translation_status(translation_id)

    async def get_rewards(self) -> Dict:
        return await self.blockchain.get_rewards(self.account)

    async def get_my_translations(self) -> List[Dict]:
        """Get all translations submitted by this translator"""
        # Get all translations from blockchain
        all_translations = await self.blockchain.get_all_translations()
        
        # Filter translations by translator address
        my_translations = [
            translation for translation in all_translations
            if translation["translator_address"].lower() == self.account.lower()
        ]
        
        # Sort by creation date, newest first
        my_translations.sort(key=lambda x: x["created_at"], reverse=True)
        
        return my_translations

    async def get_validation_history(self, translation_id: str) -> List[Dict]:
        """Get validation history for a specific translation"""
        # Get all validations from blockchain
        all_validations = await self.blockchain.get_all_validations()
        
        # Filter validations by translation ID
        translation_validations = [
            validation for validation in all_validations
            if validation["translation_id"] == translation_id
        ]
        
        # Sort by creation date, newest first
        translation_validations.sort(key=lambda x: x["created_at"], reverse=True)
        
        return translation_validations

    async def get_statistics(self) -> Dict:
        """Get translator statistics"""
        # Get all translations and validations
        my_translations = await self.get_my_translations()
        
        # Calculate statistics
        total_translations = len(my_translations)
        total_validations = sum(len(await self.get_validation_history(t["id"])) for t in my_translations)
        
        # Calculate average validation score
        validation_scores = []
        for translation in my_translations:
            validations = await self.get_validation_history(translation["id"])
            if validations:
                avg_score = sum(v["score"] for v in validations) / len(validations)
                validation_scores.append(avg_score)
        
        avg_validation_score = sum(validation_scores) / len(validation_scores) if validation_scores else 0
        
        # Get rewards
        rewards = await self.get_rewards()
        
        return {
            "total_translations": total_translations,
            "total_validations": total_validations,
            "average_validation_score": avg_validation_score,
            "total_rewards": rewards.get("total", 0),
            "pending_rewards": rewards.get("pending", 0),
            "claimed_rewards": rewards.get("claimed", 0)
        } 