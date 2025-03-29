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

async function upgradeToken() {
    console.log("\nUpgrading token contract...");
    
    const { token } = await getContracts();
    const [signer] = await hre.ethers.getSigners();
    
    // Check if signer is owner
    const owner = await token.owner();
    if (owner.toLowerCase() !== signer.address.toLowerCase()) {
        throw new Error("Only the owner can upgrade the contract");
    }
    
    // Deploy new implementation
    const Token = await hre.ethers.getContractFactory("PUMPFUNToken");
    const newToken = await Token.deploy(
        config.token.initialSupply,
        config.token.burnRate,
        config.token.minimumStake
    );
    await newToken.deployed();
    
    console.log("New token implementation deployed at:", newToken.address);
    
    // Verify on Etherscan if API key is provided
    if (config.etherscan.apiKey) {
        console.log("\nVerifying contract on Etherscan...");
        await hre.run("verify:verify", {
            address: newToken.address,
            constructorArguments: [
                config.token.initialSupply,
                config.token.burnRate,
                config.token.minimumStake
            ]
        });
        console.log("Contract verified successfully!");
    }
    
    return newToken;
}

async function upgradeTranslation() {
    console.log("\nUpgrading translation contract...");
    
    const { translation } = await getContracts();
    const [signer] = await hre.ethers.getSigners();
    
    // Check if signer is owner
    const owner = await translation.owner();
    if (owner.toLowerCase() !== signer.address.toLowerCase()) {
        throw new Error("Only the owner can upgrade the contract");
    }
    
    // Deploy new implementation
    const Translation = await hre.ethers.getContractFactory("TranslationContract");
    const newTranslation = await Translation.deploy(
        config.translation.minimumStake,
        config.translation.minimumValidations,
        config.translation.minimumScore,
        config.translation.baseReward,
        config.translation.goodScoreMultiplier,
        config.translation.poorScoreMultiplier,
        config.translation.validatorRewardMultiplier
    );
    await newTranslation.deployed();
    
    console.log("New translation implementation deployed at:", newTranslation.address);
    
    // Verify on Etherscan if API key is provided
    if (config.etherscan.apiKey) {
        console.log("\nVerifying contract on Etherscan...");
        await hre.run("verify:verify", {
            address: newTranslation.address,
            constructorArguments: [
                config.translation.minimumStake,
                config.translation.minimumValidations,
                config.translation.minimumScore,
                config.translation.baseReward,
                config.translation.goodScoreMultiplier,
                config.translation.poorScoreMultiplier,
                config.translation.validatorRewardMultiplier
            ]
        });
        console.log("Contract verified successfully!");
    }
    
    return newTranslation;
}

async function updateTokenParameters() {
    console.log("\nUpdating token parameters...");
    
    const { token } = await getContracts();
    const [signer] = await hre.ethers.getSigners();
    
    // Check if signer is owner
    const owner = await token.owner();
    if (owner.toLowerCase() !== signer.address.toLowerCase()) {
        throw new Error("Only the owner can update parameters");
    }
    
    // Update burn rate
    const currentBurnRate = await token.burnRate();
    if (currentBurnRate !== config.token.burnRate) {
        console.log("Updating burn rate...");
        const tx = await token.setBurnRate(config.token.burnRate);
        await tx.wait();
        console.log("Burn rate updated to:", config.token.burnRate, "%");
    }
    
    // Update minimum stake
    const currentMinStake = await token.minimumStake();
    if (currentMinStake !== config.token.minimumStake) {
        console.log("Updating minimum stake...");
        const tx = await token.setMinimumStake(config.token.minimumStake);
        await tx.wait();
        console.log("Minimum stake updated to:", hre.ethers.utils.formatEther(config.token.minimumStake), "tokens");
    }
}

async function updateTranslationParameters() {
    console.log("\nUpdating translation parameters...");
    
    const { translation } = await getContracts();
    const [signer] = await hre.ethers.getSigners();
    
    // Check if signer is owner
    const owner = await translation.owner();
    if (owner.toLowerCase() !== signer.address.toLowerCase()) {
        throw new Error("Only the owner can update parameters");
    }
    
    // Update minimum stake
    const currentMinStake = await translation.minimumStake();
    if (currentMinStake !== config.translation.minimumStake) {
        console.log("Updating minimum stake...");
        const tx = await translation.setMinimumStake(config.translation.minimumStake);
        await tx.wait();
        console.log("Minimum stake updated to:", hre.ethers.utils.formatEther(config.translation.minimumStake), "tokens");
    }
    
    // Update minimum validations
    const currentMinValidations = await translation.minimumValidations();
    if (currentMinValidations !== config.translation.minimumValidations) {
        console.log("Updating minimum validations...");
        const tx = await translation.setMinimumValidations(config.translation.minimumValidations);
        await tx.wait();
        console.log("Minimum validations updated to:", config.translation.minimumValidations);
    }
    
    // Update minimum score
    const currentMinScore = await translation.minimumScore();
    if (currentMinScore !== config.translation.minimumScore) {
        console.log("Updating minimum score...");
        const tx = await translation.setMinimumScore(config.translation.minimumScore);
        await tx.wait();
        console.log("Minimum score updated to:", config.translation.minimumScore, "%");
    }
    
    // Update base reward
    const currentBaseReward = await translation.baseReward();
    if (currentBaseReward !== config.translation.baseReward) {
        console.log("Updating base reward...");
        const tx = await translation.setBaseReward(config.translation.baseReward);
        await tx.wait();
        console.log("Base reward updated to:", hre.ethers.utils.formatEther(config.translation.baseReward), "tokens");
    }
    
    // Update multipliers
    const currentGoodMultiplier = await translation.goodScoreMultiplier();
    if (currentGoodMultiplier !== config.translation.goodScoreMultiplier) {
        console.log("Updating good score multiplier...");
        const tx = await translation.setGoodScoreMultiplier(config.translation.goodScoreMultiplier);
        await tx.wait();
        console.log("Good score multiplier updated to:", config.translation.goodScoreMultiplier);
    }
    
    const currentPoorMultiplier = await translation.poorScoreMultiplier();
    if (currentPoorMultiplier !== config.translation.poorScoreMultiplier) {
        console.log("Updating poor score multiplier...");
        const tx = await translation.setPoorScoreMultiplier(config.translation.poorScoreMultiplier);
        await tx.wait();
        console.log("Poor score multiplier updated to:", config.translation.poorScoreMultiplier);
    }
    
    const currentValidatorMultiplier = await translation.validatorRewardMultiplier();
    if (currentValidatorMultiplier !== config.translation.validatorRewardMultiplier) {
        console.log("Updating validator reward multiplier...");
        const tx = await translation.setValidatorRewardMultiplier(config.translation.validatorRewardMultiplier);
        await tx.wait();
        console.log("Validator reward multiplier updated to:", config.translation.validatorRewardMultiplier);
    }
}

async function updateDeploymentInfo(token, translation) {
    const deploymentsPath = path.join(__dirname, "..", "deployments.json");
    const network = await hre.ethers.provider.getNetwork();
    
    const deployments = {
        network: network.name,
        chainId: network.chainId,
        timestamp: Date.now(),
        token: {
            address: token.address,
            implementation: token.address
        },
        translation: {
            address: translation.address,
            implementation: translation.address
        }
    };
    
    fs.writeFileSync(deploymentsPath, JSON.stringify(deployments, null, 2));
    console.log("\nDeployment info updated successfully!");
}

async function main() {
    try {
        const network = await hre.ethers.provider.getNetwork();
        console.log(`\nManaging contracts on network: ${network.name} (chainId: ${network.chainId})`);
        
        const command = process.argv[2];
        const args = process.argv.slice(3);
        
        switch (command) {
            case "upgrade-token":
                const newToken = await upgradeToken();
                const { translation } = await getContracts();
                await updateDeploymentInfo(newToken, translation);
                break;
                
            case "upgrade-translation":
                const { token } = await getContracts();
                const newTranslation = await upgradeTranslation();
                await updateDeploymentInfo(token, newTranslation);
                break;
                
            case "update-token":
                await updateTokenParameters();
                break;
                
            case "update-translation":
                await updateTranslationParameters();
                break;
                
            case "update-all":
                await updateTokenParameters();
                await updateTranslationParameters();
                break;
                
            default:
                console.error("Invalid command. Available commands:");
                console.error("  upgrade-token     - Upgrade token contract");
                console.error("  upgrade-translation - Upgrade translation contract");
                console.error("  update-token      - Update token parameters");
                console.error("  update-translation - Update translation parameters");
                console.error("  update-all        - Update all parameters");
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