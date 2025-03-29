"""
Blockchain smart contract interfaces for OpenTranslate
"""

from typing import Dict, List, Optional, Union
from uuid import UUID
import json
from pathlib import Path
from web3 import Web3
from eth_typing import Address
from eth_utils import to_checksum_address

from ..config.default import settings
from ..utils.exceptions import BlockchainError

class BaseContract:
    """Base class for smart contract interactions"""
    
    def __init__(self, address: str, abi_path: Optional[str] = None):
        """
        Initialize contract interface
        
        Args:
            address: Contract address
            abi_path: Path to contract ABI file
        """
        self.w3 = Web3(Web3.HTTPProvider(settings.BLOCKCHAIN_PROVIDER))
        self.address = to_checksum_address(address)
        
        # Load contract ABI
        if abi_path:
            with open(abi_path) as f:
                self.abi = json.load(f)
        else:
            contract_dir = Path(__file__).parent / "abi"
            abi_file = contract_dir / f"{self.__class__.__name__}.json"
            with open(abi_file) as f:
                self.abi = json.load(f)
                
        # Initialize contract
        self.contract = self.w3.eth.contract(
            address=self.address,
            abi=self.abi
        )
        
    def _send_transaction(self, function, *args, **kwargs):
        """Helper method to send transactions"""
        try:
            # Build transaction
            tx = function(*args, **kwargs).build_transaction({
                "chainId": settings.CHAIN_ID,
                "gas": settings.GAS_LIMIT,
                "gasPrice": self.w3.eth.gas_price,
                "nonce": self.w3.eth.get_transaction_count(
                    self.w3.eth.default_account
                ),
            })
            
            # Sign and send transaction
            signed_tx = self.w3.eth.account.sign_transaction(
                tx,
                private_key=settings.PRIVATE_KEY
            )
            tx_hash = self.w3.eth.send_raw_transaction(
                signed_tx.rawTransaction
            )
            
            # Wait for transaction receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            return receipt
            
        except Exception as e:
            raise BlockchainError(f"Transaction failed: {e}")

class TranslationContract(BaseContract):
    """Interface for the translation smart contract"""
    
    def record_translation(
        self,
        task_id: UUID,
        source_text: Union[str, List[str]],
        translation: Union[str, List[str]],
        source_lang: str,
        target_lang: str,
        domain: str,
        score: float
    ) -> str:
        """
        Record a translation on the blockchain
        
        Args:
            task_id: Unique task identifier
            source_text: Original text
            translation: Translated text
            source_lang: Source language code
            target_lang: Target language code
            domain: Text domain/field
            score: Translation quality score
            
        Returns:
            Transaction hash
        """
        try:
            receipt = self._send_transaction(
                self.contract.functions.recordTranslation,
                str(task_id),
                source_text if isinstance(source_text, str) else json.dumps(source_text),
                translation if isinstance(translation, str) else json.dumps(translation),
                source_lang,
                target_lang,
                domain,
                int(score * 100)  # Convert to percentage
            )
            return receipt["transactionHash"].hex()
            
        except Exception as e:
            raise BlockchainError(f"Failed to record translation: {e}")
            
    def get_translation(self, task_id: UUID) -> Dict:
        """Get translation record from blockchain"""
        try:
            result = self.contract.functions.getTranslation(
                str(task_id)
            ).call()
            
            return {
                "task_id": task_id,
                "source_text": json.loads(result[0]) if "[" in result[0] else result[0],
                "translation": json.loads(result[1]) if "[" in result[1] else result[1],
                "source_lang": result[2],
                "target_lang": result[3],
                "domain": result[4],
                "score": result[5] / 100,  # Convert from percentage
                "timestamp": result[6],
                "translator": result[7]
            }
            
        except Exception as e:
            raise BlockchainError(f"Failed to get translation: {e}")
            
    def get_translator_stats(self, address: Address) -> Dict:
        """Get translator statistics from blockchain"""
        try:
            result = self.contract.functions.getTranslatorStats(
                address
            ).call()
            
            return {
                "address": address,
                "total_translations": result[0],
                "total_score": result[1] / 100,
                "reputation": result[2] / 100,
                "stake": result[3],
                "rewards": result[4]
            }
            
        except Exception as e:
            raise BlockchainError(f"Failed to get translator stats: {e}")
            
    def stake_tokens(self, amount: int) -> str:
        """Stake tokens for translation work"""
        try:
            receipt = self._send_transaction(
                self.contract.functions.stakeTokens,
                amount
            )
            return receipt["transactionHash"].hex()
            
        except Exception as e:
            raise BlockchainError(f"Failed to stake tokens: {e}")
            
    def withdraw_stake(self, amount: int) -> str:
        """Withdraw staked tokens"""
        try:
            receipt = self._send_transaction(
                self.contract.functions.withdrawStake,
                amount
            )
            return receipt["transactionHash"].hex()
            
        except Exception as e:
            raise BlockchainError(f"Failed to withdraw stake: {e}")
            
    def claim_rewards(self) -> str:
        """Claim earned rewards"""
        try:
            receipt = self._send_transaction(
                self.contract.functions.claimRewards
            )
            return receipt["transactionHash"].hex()
            
        except Exception as e:
            raise BlockchainError(f"Failed to claim rewards: {e}")

class PUMPFUNToken(BaseContract):
    """Interface for the PUMPFUN token contract"""
    
    def balance_of(self, address: Address) -> int:
        """Get token balance of address"""
        try:
            return self.contract.functions.balanceOf(address).call()
        except Exception as e:
            raise BlockchainError(f"Failed to get balance: {e}")
            
    def approve(self, spender: Address, amount: int) -> str:
        """Approve spender to spend tokens"""
        try:
            receipt = self._send_transaction(
                self.contract.functions.approve,
                spender,
                amount
            )
            return receipt["transactionHash"].hex()
            
        except Exception as e:
            raise BlockchainError(f"Failed to approve tokens: {e}")
            
    def transfer(self, recipient: Address, amount: int) -> str:
        """Transfer tokens to recipient"""
        try:
            receipt = self._send_transaction(
                self.contract.functions.transfer,
                recipient,
                amount
            )
            return receipt["transactionHash"].hex()
            
        except Exception as e:
            raise BlockchainError(f"Failed to transfer tokens: {e}")
            
    def total_supply(self) -> int:
        """Get total token supply"""
        try:
            return self.contract.functions.totalSupply().call()
        except Exception as e:
            raise BlockchainError(f"Failed to get total supply: {e}")
            
    def burn_rate(self) -> float:
        """Get current token burn rate"""
        try:
            return self.contract.functions.burnRate().call() / 10000  # Convert from basis points
        except Exception as e:
            raise BlockchainError(f"Failed to get burn rate: {e}") 