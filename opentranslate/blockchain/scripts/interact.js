const hre = require("hardhat");
const fs = require("fs");
const path = require("path");
const config = require("../config/contracts");

async function getContracts() {
    const deploymentsPath = path.join(__dirname, "..", "deployments.json");
    if (!fs.existsSync(deploymentsPath)) {
        throw new Error("No deployment info found. Please deploy contracts first.");
    }

    const deployments = JSON.parse(fs.readFileSync(deploymentsPath));
    const network = await hre.ethers.provider.getNetwork();
    
    if (deployments.network !== network.name || deployments.chainId !== network.chainId) {
        throw new Error("Network mismatch. Please ensure you're on the correct network.");
    }

    const token = await hre.ethers.getContractAt(
        "PUMPFUNToken",
        deployments.token.address
    );
    const translation = await hre.ethers.getContractAt(
        "TranslationContract",
        deployments.translation.address
    );

    return { token, translation };
}

async function getContractInfo() {
    console.log("\nContract Information:");
    
    const { token, translation } = await getContracts();
    
    // Token info
    const tokenName = await token.name();
    const tokenSymbol = await token.symbol();
    const totalSupply = await token.totalSupply();
    const burnRate = await token.burnRate();
    const minimumStake = await token.minimumStake();
    
    console.log("\nToken Contract:");
    console.log("Name:", tokenName);
    console.log("Symbol:", tokenSymbol);
    console.log("Total Supply:", hre.ethers.utils.formatEther(totalSupply), "tokens");
    console.log("Burn Rate:", burnRate, "%");
    console.log("Minimum Stake:", hre.ethers.utils.formatEther(minimumStake), "tokens");
    
    // Translation contract info
    const translationMinStake = await translation.minimumStake();
    const translationMinValidations = await translation.minimumValidations();
    const translationMinScore = await translation.minimumScore();
    const translationBaseReward = await translation.baseReward();
    
    console.log("\nTranslation Contract:");
    console.log("Minimum Stake:", hre.ethers.utils.formatEther(translationMinStake), "tokens");
    console.log("Minimum Validations:", translationMinValidations);
    console.log("Minimum Score:", translationMinScore, "%");
    console.log("Base Reward:", hre.ethers.utils.formatEther(translationBaseReward), "tokens");
}

async function submitTranslation(sourceText, targetText, sourceLang, targetLang, domain) {
    console.log("\nSubmitting translation...");
    
    const { token, translation } = await getContracts();
    const [signer] = await hre.ethers.getSigners();
    
    // Check stake
    const stakedBalance = await token.stakedBalance(signer.address);
    const minimumStake = await translation.minimumStake();
    
    if (stakedBalance.lt(minimumStake)) {
        throw new Error("Insufficient stake. Please stake more tokens first.");
    }
    
    // Submit translation
    const tx = await translation
        .connect(signer)
        .submitTranslation(sourceText, targetText, sourceLang, targetLang, domain);
    
    const receipt = await tx.wait();
    const translationId = await translation.getTranslationCount();
    
    console.log("Translation submitted successfully!");
    console.log("Translation ID:", translationId);
    console.log("Transaction hash:", receipt.transactionHash);
    
    return translationId;
}

async function validateTranslation(translationId, score, feedback) {
    console.log("\nValidating translation...");
    
    const { token, translation } = await getContracts();
    const [signer] = await hre.ethers.getSigners();
    
    // Check stake
    const stakedBalance = await token.stakedBalance(signer.address);
    const minimumStake = await translation.minimumStake();
    
    if (stakedBalance.lt(minimumStake)) {
        throw new Error("Insufficient stake. Please stake more tokens first.");
    }
    
    // Validate translation
    const tx = await translation
        .connect(signer)
        .validateTranslation(translationId, score, feedback);
    
    const receipt = await tx.wait();
    const validationCount = await translation.getValidationCount(translationId);
    
    console.log("Translation validated successfully!");
    console.log("Current validation count:", validationCount);
    console.log("Transaction hash:", receipt.transactionHash);
}

async function getTranslationStatus(translationId) {
    console.log("\nTranslation Status:");
    
    const { translation } = await getContracts();
    
    const translationData = await translation.getTranslation(translationId);
    const validationCount = await translation.getValidationCount(translationId);
    const status = translationData.status;
    
    console.log("Source Text:", translationData.sourceText);
    console.log("Target Text:", translationData.targetText);
    console.log("Source Language:", translationData.sourceLang);
    console.log("Target Language:", translationData.targetLang);
    console.log("Domain:", translationData.domain);
    console.log("Translator:", translationData.translator);
    console.log("Status:", status === 0 ? "Pending" : status === 1 ? "Completed" : "Failed");
    console.log("Validation Count:", validationCount);
    
    if (translationData.averageScore > 0) {
        console.log("Average Score:", translationData.averageScore, "%");
    }
    
    // Get validations
    for (let i = 0; i < validationCount; i++) {
        const validation = await translation.getValidation(translationId, i);
        console.log(`\nValidation ${i + 1}:`);
        console.log("Validator:", validation.validator);
        console.log("Score:", validation.score, "%");
        console.log("Feedback:", validation.feedback);
    }
}

async function getStakeInfo(address) {
    console.log("\nStake Information:");
    
    const { token } = await getContracts();
    
    const stakedBalance = await token.stakedBalance(address);
    const totalBalance = await token.balanceOf(address);
    
    console.log("Address:", address);
    console.log("Staked Balance:", hre.ethers.utils.formatEther(stakedBalance), "tokens");
    console.log("Total Balance:", hre.ethers.utils.formatEther(totalBalance), "tokens");
}

async function getRewardInfo(address) {
    console.log("\nReward Information:");
    
    const { translation } = await getContracts();
    
    const translatorScore = await translation.getTranslatorScore(address);
    const validatorScore = await translation.getValidatorScore(address);
    
    console.log("Address:", address);
    console.log("Translator Score:", translatorScore);
    console.log("Validator Score:", validatorScore);
}

async function main() {
    try {
        const network = await hre.ethers.provider.getNetwork();
        console.log(`\nInteracting on network: ${network.name} (chainId: ${network.chainId})`);
        
        const command = process.argv[2];
        const args = process.argv.slice(3);
        
        switch (command) {
            case "info":
                await getContractInfo();
                break;
                
            case "submit":
                if (args.length < 5) {
                    throw new Error("Usage: submit <sourceText> <targetText> <sourceLang> <targetLang> <domain>");
                }
                await submitTranslation(args[0], args[1], args[2], args[3], args[4]);
                break;
                
            case "validate":
                if (args.length < 3) {
                    throw new Error("Usage: validate <translationId> <score> <feedback>");
                }
                await validateTranslation(
                    parseInt(args[0]),
                    parseInt(args[1]),
                    args[2]
                );
                break;
                
            case "status":
                if (args.length < 1) {
                    throw new Error("Usage: status <translationId>");
                }
                await getTranslationStatus(parseInt(args[0]));
                break;
                
            case "stake":
                if (args.length < 1) {
                    throw new Error("Usage: stake <address>");
                }
                await getStakeInfo(args[0]);
                break;
                
            case "rewards":
                if (args.length < 1) {
                    throw new Error("Usage: rewards <address>");
                }
                await getRewardInfo(args[0]);
                break;
                
            default:
                console.error("Invalid command. Available commands:");
                console.error("  info                    - Get contract information");
                console.error("  submit <text> <lang>    - Submit a translation");
                console.error("  validate <id> <score>   - Validate a translation");
                console.error("  status <id>             - Get translation status");
                console.error("  stake <address>         - Get stake information");
                console.error("  rewards <address>       - Get reward information");
                process.exit(1);
        }
        
    } catch (error) {
        console.error("\nError:", error.message);
        process.exit(1);
    }
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error(error);
        process.exit(1);
    }); 