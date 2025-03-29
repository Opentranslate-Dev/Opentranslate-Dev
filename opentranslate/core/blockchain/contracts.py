"""
OpenTranslate Blockchain Contract Interface
"""

from typing import Dict, List, Optional
from web3 import Web3
from web3.contract import Contract
from eth_account import Account
import json

class BlockchainContracts:
    """OpenTranslate Blockchain Contract Interface"""
    
    def __init__(
        self,
        web3: Web3,
        token_address: str,
        translation_address: str,
        token_abi_path: str = "contracts/Token.json",
        translation_abi_path: str = "contracts/Translation.json",
    ):
        """
        Initialize contract interface
        
        Args:
            web3: Web3 instance
            token_address: Token contract address
            translation_address: Translation contract address
            token_abi_path: Token contract ABI file path
            translation_abi_path: Translation contract ABI file path
        """
        self.web3 = web3
        self.token_address = token_address
        self.translation_address = translation_address
        
        # Load ABI
        with open(token_abi_path) as f:
            token_abi = json.load(f)
        with open(translation_abi_path) as f:
            translation_abi = json.load(f)
            
        # Create contract instances
        self.token_contract = self.web3.eth.contract(
            address=self.web3.to_checksum_address(token_address),
            abi=token_abi
        )
        self.translation_contract = self.web3.eth.contract(
            address=self.web3.to_checksum_address(translation_address),
            abi=translation_abi
        )

    async def submit_translation(
        self,
        account: Account,
        source_text: str,
        target_text: str,
        source_lang: str,
        target_lang: str,
        domain: str
    ) -> str:
        """Submit translation to blockchain"""
        # Build translation data
        translation_data = {
            "source_text": source_text,
            "target_text": target_text,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "domain": domain,
            "translator_address": account.address,
            "status": "pending",
            "created_at": self.web3.eth.get_block('latest').timestamp,
            "updated_at": self.web3.eth.get_block('latest').timestamp
        }
        
        # Calculate content hash
        content_hash = self.web3.keccak(
            text=json.dumps(translation_data, sort_keys=True)
        ).hex()
        
        # Build transaction
        tx = self.translation_contract.functions.recordTranslation(
            content_hash,
            json.dumps(translation_data)
        ).build_transaction({
            "from": account.address,
            "nonce": self.web3.eth.get_transaction_count(account.address),
            "gas": 2000000,
            "gasPrice": self.web3.eth.gas_price,
        })
        
        # Sign and send transaction
        signed_tx = self.web3.eth.account.sign_transaction(tx, account.key)
        tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        return tx_hash.hex()

    async def validate_translation(
        self,
        account: Account,
        translation_id: str,
        score: int,
        feedback: str
    ) -> bool:
        """Validate translation"""
        # Build validation data
        validation_data = {
            "translation_id": translation_id,
            "validator_address": account.address,
            "score": score,
            "feedback": feedback,
            "created_at": self.web3.eth.get_block('latest').timestamp
        }
        
        # Build transaction
        tx = self.translation_contract.functions.validateTranslation(
            translation_id,
            score,
            feedback
        ).build_transaction({
            "from": account.address,
            "nonce": self.web3.eth.get_transaction_count(account.address),
            "gas": 2000000,
            "gasPrice": self.web3.eth.gas_price,
        })
        
        # Sign and send transaction
        signed_tx = self.web3.eth.account.sign_transaction(tx, account.key)
        tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        # Wait for transaction confirmation
        receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
        return receipt.status == 1

    async def get_translation_status(self, translation_id: str) -> Dict:
        """Get translation status"""
        try:
            data = self.translation_contract.functions.getTranslation(translation_id).call()
            return json.loads(data)
        except:
            return None

    async def get_rewards(self, account: Account) -> Dict:
        """Get account rewards"""
        address = account.address
        total_rewards = await self.token_contract.functions.balanceOf(address).call()
        pending_rewards = await self.translation_contract.functions.getPendingRewards(address).call()
        claimed_rewards = await self.translation_contract.functions.getClaimedRewards(address).call()
        
        return {
            "total": total_rewards,
            "pending": pending_rewards,
            "claimed": claimed_rewards
        }

    async def get_all_translations(self) -> List[Dict]:
        """Get all translations"""
        total_translations = await self.translation_contract.functions.getTotalTranslations().call()
        
        translations = []
        for i in range(total_translations):
            translation = await self.translation_contract.functions.getTranslation(i).call()
            translations.append({
                "id": translation[0],
                "source_text": translation[1],
                "target_text": translation[2],
                "source_lang": translation[3],
                "target_lang": translation[4],
                "domain": translation[5],
                "translator_address": translation[6],
                "status": translation[7],
                "created_at": translation[8],
                "updated_at": translation[9]
            })
        
        return translations

    async def get_all_validations(self) -> List[Dict]:
        """Get all validations"""
        total_validations = await self.translation_contract.functions.getTotalValidations().call()
        
        validations = []
        for i in range(total_validations):
            validation = await self.translation_contract.functions.getValidation(i).call()
            validations.append({
                "id": validation[0],
                "translation_id": validation[1],
                "validator_address": validation[2],
                "score": validation[3],
                "feedback": validation[4],
                "created_at": validation[5]
            })
        
        return validations

    async def get_translation_validations(self, translation_id: str) -> List[Dict]:
        """Get all validations for a specific translation"""
        total_validations = await self.translation_contract.functions.getTranslationValidationsCount(translation_id).call()
        
        validations = []
        for i in range(total_validations):
            validation = await self.translation_contract.functions.getTranslationValidation(translation_id, i).call()
            validations.append({
                "id": validation[0],
                "validator_address": validation[1],
                "score": validation[2],
                "feedback": validation[3],
                "created_at": validation[4]
            })
        
        return validations

    async def get_translator_statistics(self, address: str) -> Dict:
        """Get translator statistics"""
        stats = await self.translation_contract.functions.getTranslatorStats(address).call()
        return {
            "total_translations": stats[0],
            "total_validations": stats[1],
            "average_validation_score": stats[2],
            "total_rewards": stats[3],
            "pending_rewards": stats[4],
            "claimed_rewards": stats[5]
        }

    async def get_translator_count(self, address: str) -> int:
        """Get translator translation count"""
        return await self.translation_contract.functions.getTranslatorCount(
            self.web3.to_checksum_address(address)
        ).call()

    async def get_reputation_score(self, address: str) -> float:
        """Get translator reputation score"""
        return await self.translation_contract.functions.getReputationScore(
            self.web3.to_checksum_address(address)
        ).call() / 100  # Convert to 0-1 range

    async def get_validation_rate(self, address: str) -> float:
        """Get translator validation rate"""
        return await self.translation_contract.functions.getValidationRate(
            self.web3.to_checksum_address(address)
        ).call() / 100  # Convert to 0-1 range

    async def get_total_translations(self) -> int:
        """Get total translation count"""
        return await self.translation_contract.functions.getTotalTranslations().call()

    async def get_active_translators(self) -> int:
        """Get active translator count"""
        return await self.translation_contract.functions.getActiveTranslators().call()

    async def get_average_validation_score(self) -> float:
        """Get average validation score"""
        return await self.translation_contract.functions.getAverageValidationScore().call() / 100  # Convert to 0-1 range 