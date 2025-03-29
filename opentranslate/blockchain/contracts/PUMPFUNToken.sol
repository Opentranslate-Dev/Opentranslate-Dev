// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/Pausable.sol";

/**
 * @title PUMPFUN Token
 * @dev Implementation of the PUMPFUN token for OpenTranslate
 */
contract PUMPFUNToken is ERC20, Ownable, Pausable {
    // Token parameters
    uint256 public constant INITIAL_SUPPLY = 1000000000 * 10**18; // 1 billion tokens
    uint256 public constant BURN_RATE = 2; // 2% burn rate
    uint256 public constant MIN_STAKE = 1000 * 10**18; // 1000 tokens minimum stake
    
    // Mapping for staked tokens
    mapping(address => uint256) public stakedTokens;
    
    // Events
    event TokensStaked(address indexed user, uint256 amount);
    event TokensUnstaked(address indexed user, uint256 amount);
    event TokensBurned(address indexed user, uint256 amount);
    event RewardsDistributed(address indexed user, uint256 amount);
    
    constructor() ERC20("PUMPFUN Token", "PUMPFUN") {
        _mint(msg.sender, INITIAL_SUPPLY);
    }
    
    /**
     * @dev Stake tokens for translation activities
     * @param amount Amount of tokens to stake
     */
    function stake(uint256 amount) external whenNotPaused {
        require(amount >= MIN_STAKE, "Amount below minimum stake");
        require(balanceOf(msg.sender) >= amount, "Insufficient balance");
        
        _transfer(msg.sender, address(this), amount);
        stakedTokens[msg.sender] += amount;
        
        emit TokensStaked(msg.sender, amount);
    }
    
    /**
     * @dev Unstake tokens
     * @param amount Amount of tokens to unstake
     */
    function unstake(uint256 amount) external whenNotPaused {
        require(stakedTokens[msg.sender] >= amount, "Insufficient staked tokens");
        
        stakedTokens[msg.sender] -= amount;
        _transfer(address(this), msg.sender, amount);
        
        emit TokensUnstaked(msg.sender, amount);
    }
    
    /**
     * @dev Burn tokens for translation quality assurance
     * @param amount Amount of tokens to burn
     */
    function burn(uint256 amount) external whenNotPaused {
        require(balanceOf(msg.sender) >= amount, "Insufficient balance");
        
        _burn(msg.sender, amount);
        emit TokensBurned(msg.sender, amount);
    }
    
    /**
     * @dev Distribute rewards to translators
     * @param user Address of the translator
     * @param amount Amount of tokens to distribute
     */
    function distributeRewards(address user, uint256 amount) external onlyOwner whenNotPaused {
        require(amount <= balanceOf(address(this)), "Insufficient contract balance");
        
        _transfer(address(this), user, amount);
        emit RewardsDistributed(user, amount);
    }
    
    /**
     * @dev Pause token operations
     */
    function pause() external onlyOwner {
        _pause();
    }
    
    /**
     * @dev Unpause token operations
     */
    function unpause() external onlyOwner {
        _unpause();
    }
    
    /**
     * @dev Override transfer to implement burn rate
     */
    function transfer(address to, uint256 amount) public virtual override returns (bool) {
        uint256 burnAmount = (amount * BURN_RATE) / 100;
        uint256 transferAmount = amount - burnAmount;
        
        _burn(msg.sender, burnAmount);
        _transfer(msg.sender, to, transferAmount);
        
        return true;
    }
    
    /**
     * @dev Get staked token balance
     */
    function getStakedBalance(address user) external view returns (uint256) {
        return stakedTokens[user];
    }
} 