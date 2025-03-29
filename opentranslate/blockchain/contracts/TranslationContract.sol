// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "./PUMPFUNToken.sol";

/**
 * @title Translation Contract
 * @dev Manages translations and quality assurance on the blockchain
 */
contract TranslationContract is Ownable, Pausable, ReentrancyGuard {
    // Token contract
    PUMPFUNToken public token;
    
    // Translation struct
    struct Translation {
        string sourceText;
        string targetText;
        string sourceLang;
        string targetLang;
        string domain;
        address translator;
        uint256 timestamp;
        uint256 score;
        bool validated;
        bool completed;
    }
    
    // Validation struct
    struct Validation {
        address validator;
        uint256 score;
        string feedback;
        uint256 timestamp;
    }
    
    // Contract parameters
    uint256 public constant MIN_VALIDATIONS = 3;
    uint256 public constant MIN_SCORE = 70;
    uint256 public constant REWARD_MULTIPLIER = 100; // 100 tokens per point
    
    // Mappings
    mapping(bytes32 => Translation) public translations;
    mapping(bytes32 => Validation[]) public validations;
    mapping(address => uint256) public translatorScores;
    mapping(address => uint256) public validatorScores;
    
    // Events
    event TranslationSubmitted(
        bytes32 indexed translationId,
        address indexed translator,
        string sourceLang,
        string targetLang
    );
    event TranslationValidated(
        bytes32 indexed translationId,
        address indexed validator,
        uint256 score
    );
    event TranslationCompleted(
        bytes32 indexed translationId,
        uint256 finalScore
    );
    event RewardsDistributed(
        address indexed translator,
        address indexed validator,
        uint256 amount
    );
    
    constructor(address _token) {
        token = PUMPFUNToken(_token);
    }
    
    /**
     * @dev Submit a new translation
     */
    function submitTranslation(
        string memory sourceText,
        string memory targetText,
        string memory sourceLang,
        string memory targetLang,
        string memory domain
    ) external whenNotPaused nonReentrant {
        require(token.getStakedBalance(msg.sender) > 0, "No staked tokens");
        
        bytes32 translationId = keccak256(
            abi.encodePacked(
                sourceText,
                targetText,
                sourceLang,
                targetLang,
                domain,
                msg.sender,
                block.timestamp
            )
        );
        
        translations[translationId] = Translation({
            sourceText: sourceText,
            targetText: targetText,
            sourceLang: sourceLang,
            targetLang: targetLang,
            domain: domain,
            translator: msg.sender,
            timestamp: block.timestamp,
            score: 0,
            validated: false,
            completed: false
        });
        
        emit TranslationSubmitted(
            translationId,
            msg.sender,
            sourceLang,
            targetLang
        );
    }
    
    /**
     * @dev Validate a translation
     */
    function validateTranslation(
        bytes32 translationId,
        uint256 score,
        string memory feedback
    ) external whenNotPaused nonReentrant {
        require(token.getStakedBalance(msg.sender) > 0, "No staked tokens");
        require(!translations[translationId].completed, "Translation completed");
        require(score >= 0 && score <= 100, "Invalid score");
        
        validations[translationId].push(Validation({
            validator: msg.sender,
            score: score,
            feedback: feedback,
            timestamp: block.timestamp
        }));
        
        emit TranslationValidated(translationId, msg.sender, score);
        
        // Check if enough validations
        if (validations[translationId].length >= MIN_VALIDATIONS) {
            _completeTranslation(translationId);
        }
    }
    
    /**
     * @dev Complete translation and distribute rewards
     */
    function _completeTranslation(bytes32 translationId) internal {
        Translation storage translation = translations[translationId];
        Validation[] storage validationList = validations[translationId];
        
        // Calculate average score
        uint256 totalScore = 0;
        for (uint256 i = 0; i < validationList.length; i++) {
            totalScore += validationList[i].score;
        }
        uint256 averageScore = totalScore / validationList.length;
        
        translation.score = averageScore;
        translation.validated = true;
        translation.completed = true;
        
        emit TranslationCompleted(translationId, averageScore);
        
        // Distribute rewards if score is good enough
        if (averageScore >= MIN_SCORE) {
            uint256 reward = averageScore * REWARD_MULTIPLIER;
            
            // Distribute to translator
            token.distributeRewards(translation.translator, reward);
            translatorScores[translation.translator] += reward;
            
            // Distribute to validators
            uint256 validatorReward = reward / validationList.length;
            for (uint256 i = 0; i < validationList.length; i++) {
                token.distributeRewards(
                    validationList[i].validator,
                    validatorReward
                );
                validatorScores[validationList[i].validator] += validatorReward;
            }
            
            emit RewardsDistributed(
                translation.translator,
                msg.sender,
                reward
            );
        }
    }
    
    /**
     * @dev Get translation details
     */
    function getTranslation(bytes32 translationId)
        external
        view
        returns (
            string memory sourceText,
            string memory targetText,
            string memory sourceLang,
            string memory targetLang,
            string memory domain,
            address translator,
            uint256 timestamp,
            uint256 score,
            bool validated,
            bool completed
        )
    {
        Translation storage translation = translations[translationId];
        return (
            translation.sourceText,
            translation.targetText,
            translation.sourceLang,
            translation.targetLang,
            translation.domain,
            translation.translator,
            translation.timestamp,
            translation.score,
            translation.validated,
            translation.completed
        );
    }
    
    /**
     * @dev Get validation count
     */
    function getValidationCount(bytes32 translationId)
        external
        view
        returns (uint256)
    {
        return validations[translationId].length;
    }
    
    /**
     * @dev Get translator score
     */
    function getTranslatorScore(address translator)
        external
        view
        returns (uint256)
    {
        return translatorScores[translator];
    }
    
    /**
     * @dev Get validator score
     */
    function getValidatorScore(address validator)
        external
        view
        returns (uint256)
    {
        return validatorScores[validator];
    }
    
    /**
     * @dev Pause contract operations
     */
    function pause() external onlyOwner {
        _pause();
    }
    
    /**
     * @dev Unpause contract operations
     */
    function unpause() external onlyOwner {
        _unpause();
    }
} 