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

async function setupTreasury() {
    console.log("\nSetting up treasury...");
    
    const { token, translation } = await getContracts();
    const [signer] = await hre.ethers.getSigners();
    
    // Check if signer is owner
    const owner = await token.owner();
    if (owner.toLowerCase() !== signer.address.toLowerCase()) {
        throw new Error("Only the owner can set up treasury");
    }
    
    // Transfer tokens to treasury
    const treasuryAmount = hre.ethers.utils.parseEther(config.treasury.initialAmount);
    const tx1 = await token.transfer(config.treasury.address, treasuryAmount);
    await tx1.wait();
    
    console.log("Transferred", config.treasury.initialAmount, "tokens to treasury");
    
    // Transfer tokens to translation contract for rewards
    const rewardAmount = hre.ethers.utils.parseEther(config.treasury.rewardPool);
    const tx2 = await token.transfer(translation.address, rewardAmount);
    await tx2.wait();
    
    console.log("Transferred", config.treasury.rewardPool, "tokens to translation contract");
}

async function distributeRewards() {
    console.log("\nDistributing rewards...");
    
    const { token, translation } = await getContracts();
    const [signer] = await hre.ethers.getSigners();
    
    // Check if signer is owner
    const owner = await token.owner();
    if (owner.toLowerCase() !== signer.address.toLowerCase()) {
        throw new Error("Only the owner can distribute rewards");
    }
    
    // Get pending rewards
    const pendingRewards = await translation.getPendingRewards();
    if (pendingRewards.isZero()) {
        console.log("No pending rewards to distribute");
        return;
    }
    
    // Distribute rewards
    const tx = await translation.distributeRewards();
    const receipt = await tx.wait();
    
    console.log("Rewards distributed successfully!");
    console.log("Transaction hash:", receipt.transactionHash);
}

async function replenishRewardPool() {
    console.log("\nReplenishing reward pool...");
    
    const { token, translation } = await getContracts();
    const [signer] = await hre.ethers.getSigners();
    
    // Check if signer is owner
    const owner = await token.owner();
    if (owner.toLowerCase() !== signer.address.toLowerCase()) {
        throw new Error("Only the owner can replenish reward pool");
    }
    
    // Get current reward pool balance
    const currentBalance = await token.balanceOf(translation.address);
    const targetBalance = hre.ethers.utils.parseEther(config.treasury.rewardPool);
    
    if (currentBalance.gte(targetBalance)) {
        console.log("Reward pool is already at target balance");
        return;
    }
    
    // Calculate amount to transfer
    const transferAmount = targetBalance.sub(currentBalance);
    
    // Transfer tokens from treasury to translation contract
    const tx = await token.transfer(translation.address, transferAmount);
    const receipt = await tx.wait();
    
    console.log("Reward pool replenished successfully!");
    console.log("Added", hre.ethers.utils.formatEther(transferAmount), "tokens");
    console.log("Transaction hash:", receipt.transactionHash);
}

async function getTreasuryInfo() {
    console.log("\nTreasury Information:");
    
    const { token, translation } = await getContracts();
    
    // Get treasury balance
    const treasuryBalance = await token.balanceOf(config.treasury.address);
    console.log("Treasury Balance:", hre.ethers.utils.formatEther(treasuryBalance), "tokens");
    
    // Get reward pool balance
    const rewardPoolBalance = await token.balanceOf(translation.address);
    console.log("Reward Pool Balance:", hre.ethers.utils.formatEther(rewardPoolBalance), "tokens");
    
    // Get pending rewards
    const pendingRewards = await translation.getPendingRewards();
    console.log("Pending Rewards:", hre.ethers.utils.formatEther(pendingRewards), "tokens");
    
    // Get total distributed rewards
    const totalRewards = await translation.getTotalRewards();
    console.log("Total Distributed Rewards:", hre.ethers.utils.formatEther(totalRewards), "tokens");
}

async function main() {
    try {
        const network = await hre.ethers.provider.getNetwork();
        console.log(`\nManaging treasury on network: ${network.name} (chainId: ${network.chainId})`);
        
        const command = process.argv[2];
        const args = process.argv.slice(3);
        
        switch (command) {
            case "setup":
                await setupTreasury();
                break;
                
            case "distribute":
                await distributeRewards();
                break;
                
            case "replenish":
                await replenishRewardPool();
                break;
                
            case "info":
                await getTreasuryInfo();
                break;
                
            default:
                console.error("Invalid command. Available commands:");
                console.error("  setup      - Set up initial treasury and reward pool");
                console.error("  distribute - Distribute pending rewards");
                console.error("  replenish  - Replenish reward pool");
                console.error("  info       - Get treasury information");
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