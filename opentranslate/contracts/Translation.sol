// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

/**
 * @title Translation Contract
 * @dev Manages translation records and rewards on the blockchain
 */
contract Translation is Ownable, ReentrancyGuard {
    IERC20 public pumpfunToken;
    
    struct TranslationRecord {
        string id;
        address translator;
        string sourceLanguage;
        string targetLanguage;
        string domain;
        uint256 timestamp;
        uint256 reward;
        bool validated;
        uint256 score;
    }
    
    struct Translator {
        address addr;
        uint256 stake;
        uint256 totalRewards;
        uint256 reputation;
        bool isActive;
    }
    
    mapping(string => TranslationRecord) public translations;
    mapping(address => Translator) public translators;
    
    uint256 public minStake = 1000 * 10**18; // 1000 PUMPFUN
    uint256 public baseReward = 10 * 10**18; // 10 PUMPFUN
    
    event TranslationRecorded(
        string indexed id,
        address indexed translator,
        string sourceLanguage,
        string targetLanguage,
        uint256 timestamp
    );
    
    event TranslationValidated(
        string indexed id,
        address indexed validator,
        uint256 score,
        uint256 reward
    );
    
    event StakeDeposited(address indexed translator, uint256 amount);
    event StakeWithdrawn(address indexed translator, uint256 amount);
    event RewardClaimed(address indexed translator, uint256 amount);
    
    constructor(address _pumpfunToken) {
        pumpfunToken = IERC20(_pumpfunToken);
    }
    
    function depositStake(uint256 _amount) external {
        require(_amount >= minStake, "Stake amount too low");
        require(pumpfunToken.transferFrom(msg.sender, address(this), _amount), "Transfer failed");
        
        Translator storage translator = translators[msg.sender];
        translator.addr = msg.sender;
        translator.stake += _amount;
        translator.isActive = true;
        
        emit StakeDeposited(msg.sender, _amount);
    }
    
    function withdrawStake(uint256 _amount) external nonReentrant {
        Translator storage translator = translators[msg.sender];
        require(translator.stake >= _amount, "Insufficient stake");
        require(!translator.isActive || translator.stake - _amount >= minStake, "Must maintain min stake");
        
        translator.stake -= _amount;
        require(pumpfunToken.transfer(msg.sender, _amount), "Transfer failed");
        
        if (translator.stake < minStake) {
            translator.isActive = false;
        }
        
        emit StakeWithdrawn(msg.sender, _amount);
    }
    
    function recordTranslation(
        string calldata _id,
        string calldata _sourceLanguage,
        string calldata _targetLanguage,
        string calldata _domain
    ) external {
        require(translators[msg.sender].isActive, "Translator not active");
        require(translators[msg.sender].stake >= minStake, "Insufficient stake");
        
        translations[_id] = TranslationRecord({
            id: _id,
            translator: msg.sender,
            sourceLanguage: _sourceLanguage,
            targetLanguage: _targetLanguage,
            domain: _domain,
            timestamp: block.timestamp,
            reward: 0,
            validated: false,
            score: 0
        });
        
        emit TranslationRecorded(_id, msg.sender, _sourceLanguage, _targetLanguage, block.timestamp);
    }
    
    function validateTranslation(
        string calldata _id,
        uint256 _score
    ) external {
        require(_score <= 100, "Score must be <= 100");
        require(!translations[_id].validated, "Already validated");
        require(translations[_id].translator != msg.sender, "Cannot validate own translation");
        
        TranslationRecord storage translation = translations[_id];
        translation.validated = true;
        translation.score = _score;
        
        uint256 reward = calculateReward(_score);
        translation.reward = reward;
        
        Translator storage translator = translators[translation.translator];
        translator.totalRewards += reward;
        translator.reputation = (translator.reputation + _score) / 2;
        
        emit TranslationValidated(_id, msg.sender, _score, reward);
    }
    
    function claimRewards() external nonReentrant {
        Translator storage translator = translators[msg.sender];
        require(translator.totalRewards > 0, "No rewards to claim");
        
        uint256 amount = translator.totalRewards;
        translator.totalRewards = 0;
        
        require(pumpfunToken.transfer(msg.sender, amount), "Transfer failed");
        
        emit RewardClaimed(msg.sender, amount);
    }
    
    function calculateReward(uint256 _score) internal view returns (uint256) {
        return baseReward * _score / 100;
    }
    
    function setMinStake(uint256 _minStake) external onlyOwner {
        minStake = _minStake;
    }
    
    function setBaseReward(uint256 _baseReward) external onlyOwner {
        baseReward = _baseReward;
    }
    
    function getTranslator(address _translator) external view returns (
        uint256 stake,
        uint256 totalRewards,
        uint256 reputation,
        bool isActive
    ) {
        Translator memory translator = translators[_translator];
        return (
            translator.stake,
            translator.totalRewards,
            translator.reputation,
            translator.isActive
        );
    }
    
    function getTranslation(string calldata _id) external view returns (
        address translator,
        string memory sourceLanguage,
        string memory targetLanguage,
        string memory domain,
        uint256 timestamp,
        uint256 reward,
        bool validated,
        uint256 score
    ) {
        TranslationRecord memory translation = translations[_id];
        return (
            translation.translator,
            translation.sourceLanguage,
            translation.targetLanguage,
            translation.domain,
            translation.timestamp,
            translation.reward,
            translation.validated,
            translation.score
        );
    }
} 