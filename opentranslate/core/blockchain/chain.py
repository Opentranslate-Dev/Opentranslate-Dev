"""
OpenTranslate Blockchain Core Implementation
"""

from typing import Dict, List, Optional
from web3 import Web3
from eth_account import Account
import json
import time

from .contracts import TranslationContract
from .token import PUMPFUNToken
from ..translator.protocol import TranslationResponse

class Blockchain:
    """OpenTranslate Blockchain Integration"""
    
    def __init__(
        self,
        provider_url: str,
        contract_address: str,
        token_address: str,
        private_key: Optional[str] = None,
    ):
        """
        Initialize blockchain connection
        
        Args:
            provider_url: Web3 provider URL
            contract_address: Translation contract address
            token_address: PUMPFUN token address
            private_key: Private key (optional)
        """
        # Connect Web3
        self.w3 = Web3(Web3.HTTPProvider(provider_url))
        
        # Load contracts
        self.contract = TranslationContract(
            self.w3,
            contract_address
        )
        
        # Load token
        self.token = PUMPFUNToken(
            self.w3,
            token_address
        )
        
        # Set account
        if private_key:
            self.account = Account.from_key(private_key)
        else:
            self.account = Account.create()
            
    def record_translation(
        self,
        response: TranslationResponse,
    ) -> str:
        """
        Record translation to blockchain
        
        Args:
            response: Translation response
            
        Returns:
            Transaction hash
        """
        # Prepare data
        data = {
            "request_id": response.request_id,
            "source_lang": response.source_lang,
            "target_lang": response.target_lang,
            "model": response.model,
            "domain": response.domain,
            "priority": response.priority,
            "timestamp": int(time.time()),
            "metadata": response.metadata,
            "validation": response.validation
        }
        
        # Calculate content hash
        content_hash = self.w3.keccak(
            text=json.dumps(data, sort_keys=True)
        ).hex()
        
        # Record translation
        tx_hash = self.contract.record_translation(
            content_hash,
            data,
            self.account
        )
        
        # If validation is enabled, distribute token rewards
        if response.validation and response.validation["status"] == "validated":
            self.token.mint_reward(
                self.account.address,
                response.validation["score"],
                self.account
            )
            
        return tx_hash
        
    def get_translation(
        self,
        request_id: str,
    ) -> Optional[Dict]:
        """
        Get translation record
        
        Args:
            request_id: Request ID
            
        Returns:
            Translation record (if exists)
        """
        return self.contract.get_translation(request_id)
        
    def get_translator_stats(
        self,
        address: str,
    ) -> Dict:
        """
        Get translator statistics
        
        Args:
            address: Translator address
            
        Returns:
            Statistics
        """
        return {
            "total_translations": self.contract.get_translator_count(address),
            "token_balance": self.token.balance_of(address),
            "reputation_score": self.contract.get_reputation_score(address),
            "validation_rate": self.contract.get_validation_rate(address)
        }
        
    def get_network_stats(self) -> Dict:
        """
        Get network statistics
        
        Returns:
            Network statistics
        """
        return {
            "total_translations": self.contract.get_total_translations(),
            "active_translators": self.contract.get_active_translators(),
            "total_tokens_minted": self.token.total_supply(),
            "average_validation_score": self.contract.get_average_validation_score()
        } 