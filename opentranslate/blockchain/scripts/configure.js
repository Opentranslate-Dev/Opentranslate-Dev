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

async function updateTokenParameters() {
    console.log("\nUpdating token parameters...");
    
    const { token } = await getContracts();
    const [owner] = await hre.ethers.getSigners();
    
    // Update burn rate
    const newBurnRate = 3; // 3%
    await token.connect(owner).setBurnRate(newBurnRate);
    console.log("Burn rate updated to:", newBurnRate, "%");
    
    // Update minimum stake
    const newMinimumStake = hre.ethers.utils.parseEther("2000");
    await token.connect(owner).setMinimumStake(newMinimumStake);
    console.log("Minimum stake updated to:", hre.ethers.utils.formatEther(newMinimumStake), "tokens");
}

async function updateTranslationParameters() {
    console.log("\nUpdating translation parameters...");
    
    const { translation } = await getContracts();
    const [owner] = await hre.ethers.getSigners();
    
    // Update minimum stake
    const newMinimumStake = hre.ethers.utils.parseEther("2000");
    await translation.connect(owner).setMinimumStake(newMinimumStake);
    console.log("Minimum stake updated to:", hre.ethers.utils.formatEther(newMinimumStake), "tokens");
    
    // Update minimum validations
    const newMinimumValidations = 4;
    await translation.connect(owner).setMinimumValidations(newMinimumValidations);
    console.log("Minimum validations updated to:", newMinimumValidations);
    
    // Update minimum score
    const newMinimumScore = 75;
    await translation.connect(owner).setMinimumScore(newMinimumScore);
    console.log("Minimum score updated to:", newMinimumScore, "%");
    
    // Update base reward
    const newBaseReward = hre.ethers.utils.parseEther("200");
    await translation.connect(owner).setBaseReward(newBaseReward);
    console.log("Base reward updated to:", hre.ethers.utils.formatEther(newBaseReward), "tokens");
    
    // Update multipliers
    const newGoodScoreMultiplier = 130; // 130%
    const newPoorScoreMultiplier = 70; // 70%
    const newValidatorRewardMultiplier = 3; // 3x
    
    await translation.connect(owner).setMultipliers(
        newGoodScoreMultiplier,
        newPoorScoreMultiplier,
        newValidatorRewardMultiplier
    );
    
    console.log("Good score multiplier updated to:", newGoodScoreMultiplier, "%");
    console.log("Poor score multiplier updated to:", newPoorScoreMultiplier, "%");
    console.log("Validator reward multiplier updated to:", newValidatorRewardMultiplier, "x");
}

async function setupTreasury() {
    console.log("\nSetting up treasury...");
    
    const { token, translation } = await getContracts();
    const [owner, treasury] = await hre.ethers.getSigners();
    
    // Transfer tokens to treasury
    const treasuryAmount = hre.ethers.utils.parseEther(config.translation.treasury.initialAmount);
    await token.connect(owner).transfer(treasury.address, treasuryAmount);
    console.log("Transferred", hre.ethers.utils.formatEther(treasuryAmount), "tokens to treasury");
    
    // Transfer tokens to contract for rewards
    const contractRewards = hre.ethers.utils.parseEther(config.translation.rewards.contractAmount);
    await token.connect(owner).transfer(translation.address, contractRewards);
    console.log("Transferred", hre.ethers.utils.formatEther(contractRewards), "tokens to contract for rewards");
}

async function setupTestAccounts() {
    console.log("\nSetting up test accounts...");
    
    const { token } = await getContracts();
    const [owner, treasury, ...otherSigners] = await hre.ethers.getSigners();
    const stakeAmount = hre.ethers.utils.parseEther(config.translation.testAccounts.stakeAmount);
    const testAccounts = otherSigners.slice(0, config.translation.testAccounts.count);
    
    for (const account of testAccounts) {
        // Transfer tokens
        await token.connect(owner).transfer(account.address, stakeAmount * 2);
        
        // Stake tokens
        await token.connect(account).stake(stakeAmount);
        
        console.log("Setup test account:", account.address);
    }
}

async function main() {
    try {
        const network = await hre.ethers.provider.getNetwork();
        console.log(`\nConfiguring on network: ${network.name} (chainId: ${network.chainId})`);
        
        const command = process.argv[2];
        
        switch (command) {
            case "token":
                await updateTokenParameters();
                break;
            case "translation":
                await updateTranslationParameters();
                break;
            case "treasury":
                await setupTreasury();
                break;
            case "test":
                await setupTestAccounts();
                break;
            case "all":
                await updateTokenParameters();
                await updateTranslationParameters();
                await setupTreasury();
                await setupTestAccounts();
                break;
            default:
                console.error("Invalid command. Use: token, translation, treasury, test, or all");
                process.exit(1);
        }
        
        console.log("\nConfiguration completed successfully!");
        
    } catch (error) {
        console.error("\nConfiguration failed:", error);
        process.exit(1);
    }
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error(error);
        process.exit(1);
    }); 