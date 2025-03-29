"""
OpenTranslate PUMPFUN Token Contract
"""

from typing import Dict, Optional
from web3 import Web3
from web3.contract import Contract
from eth_account import Account
import json

class PUMPFUNToken:
    """PUMPFUN Token Contract"""
    
    def __init__(
        self,
        w3: Web3,
        address: str,
        abi_path: str = "contracts/PUMPFUN.json",
    ):
        """
        Initialize token contract
        
        Args:
            w3: Web3 instance
            address: Contract address
            abi_path: ABI file path
        """
        self.w3 = w3
        self.address = address
        
        # Load ABI
        with open(abi_path) as f:
            abi = json.load(f)
            
        # Create contract instance
        self.contract = self.w3.eth.contract(
            address=self.w3.to_checksum_address(address),
            abi=abi
        )
        
    def mint_reward(
        self,
        address: str,
        score: float,
        account: Account,
    ) -> str:
        """
        Mint reward tokens
        
        Args:
            address: Recipient address
            score: Validation score
            account: Minting account
            
        Returns:
            Transaction hash
        """
        # Calculate reward amount
        amount = int(score * 1000)  # Base reward 1000 tokens
        
        # Build transaction
        tx = self.contract.functions.mint(
            self.w3.to_checksum_address(address),
            amount
        ).build_transaction({
            "from": account.address,
            "nonce": self.w3.eth.get_transaction_count(account.address),
            "gas": 2000000,
            "gasPrice": self.w3.eth.gas_price,
        })
        
        # Sign and send transaction
        signed_tx = self.w3.eth.account.sign_transaction(tx, account.key)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        return tx_hash.hex()
        
    def balance_of(
        self,
        address: str,
    ) -> int:
        """
        Get token balance
        
        Args:
            address: Account address
            
        Returns:
            Token balance
        """
        return self.contract.functions.balanceOf(
            self.w3.to_checksum_address(address)
        ).call()
        
    def total_supply(self) -> int:
        """
        Get total supply
        
        Returns:
            Total supply
        """
        return self.contract.functions.totalSupply().call()
        
    def transfer(
        self,
        to_address: str,
        amount: int,
        account: Account,
    ) -> str:
        """
        Transfer tokens
        
        Args:
            to_address: Recipient address
            amount: Transfer amount
            account: Sending account
            
        Returns:
            Transaction hash
        """
        # Build transaction
        tx = self.contract.functions.transfer(
            self.w3.to_checksum_address(to_address),
            amount
        ).build_transaction({
            "from": account.address,
            "nonce": self.w3.eth.get_transaction_count(account.address),
            "gas": 2000000,
            "gasPrice": self.w3.eth.gas_price,
        })
        
        # Sign and send transaction
        signed_tx = self.w3.eth.account.sign_transaction(tx, account.key)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        return tx_hash.hex()
        
    def approve(
        self,
        spender: str,
        amount: int,
        account: Account,
    ) -> str:
        """
        Approve token usage
        
        Args:
            spender: Authorized address
            amount: Authorized amount
            account: Authorizing account
            
        Returns:
            Transaction hash
        """
        # Build transaction
        tx = self.contract.functions.approve(
            self.w3.to_checksum_address(spender),
            amount
        ).build_transaction({
            "from": account.address,
            "nonce": self.w3.eth.get_transaction_count(account.address),
            "gas": 2000000,
            "gasPrice": self.w3.eth.gas_price,
        })
        
        # Sign and send transaction
        signed_tx = self.w3.eth.account.sign_transaction(tx, account.key)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        return tx_hash.hex()
        
    def allowance(
        self,
        owner: str,
        spender: str,
    ) -> int:
        """
        Get authorized allowance
        
        Args:
            owner: Owner address
            spender: Authorized address
            
        Returns:
            Authorized allowance
        """
        return self.contract.functions.allowance(
            self.w3.to_checksum_address(owner),
            self.w3.to_checksum_address(spender)
        ).call() 