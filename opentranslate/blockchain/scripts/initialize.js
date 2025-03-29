const hre = require("hardhat");

async function main() {
    console.log("Initializing token distribution and contract setup...");
    
    // Load deployment info
    const deployments = require("../deployments.json");
    if (!deployments.token || !deployments.translation) {
        throw new Error("Contracts not deployed. Please deploy contracts first.");
    }
    
    // Get contract instances
    const token = await hre.ethers.getContractAt(
        "PUMPFUNToken",
        deployments.token.address
    );
    const translation = await hre.ethers.getContractAt(
        "TranslationContract",
        deployments.translation.address
    );
    
    // Get signers
    const [owner, treasury, ...otherSigners] = await hre.ethers.getSigners();
    
    // Transfer tokens to treasury
    const treasuryAmount = hre.ethers.utils.parseEther("100000000"); // 100M tokens
    console.log("Transferring tokens to treasury...");
    await token.transfer(treasury.address, treasuryAmount);
    
    // Transfer tokens to contract for rewards
    const contractRewards = hre.ethers.utils.parseEther("50000000"); // 50M tokens
    console.log("Transferring tokens to contract for rewards...");
    await token.transfer(translation.address, contractRewards);
    
    // Grant contract permission to distribute rewards
    console.log("Granting contract permission to distribute rewards...");
    await token.grantRole(await token.REWARD_DISTRIBUTOR_ROLE(), translation.address);
    
    // Set up initial staking for test accounts
    const stakeAmount = hre.ethers.utils.parseEther("1000");
    console.log("Setting up initial staking for test accounts...");
    
    for (const signer of otherSigners.slice(0, 5)) {
        // Transfer tokens
        await token.transfer(signer.address, stakeAmount * 2);
        
        // Stake tokens
        await token.connect(signer).stake(stakeAmount);
    }
    
    console.log("Initialization complete!");
    console.log("Treasury address:", treasury.address);
    console.log("Treasury balance:", hre.ethers.utils.formatEther(await token.balanceOf(treasury.address)));
    console.log("Contract rewards balance:", hre.ethers.utils.formatEther(await token.balanceOf(translation.address)));
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error(error);
        process.exit(1);
    }); 