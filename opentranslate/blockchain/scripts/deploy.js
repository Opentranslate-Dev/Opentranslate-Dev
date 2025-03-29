const hre = require("hardhat");
const fs = require("fs");
const path = require("path");
const config = require("../config/contracts");

async function deployToken() {
    console.log("\nDeploying PUMPFUN token...");
    
    // Deploy token
    const PUMPFUNToken = await hre.ethers.getContractFactory("PUMPFUNToken");
    const token = await PUMPFUNToken.deploy();
    await token.deployed();
    
    console.log("PUMPFUN token deployed to:", token.address);
    await token.deployTransaction.wait(5);
    
    return token;
}

async function deployTranslation(tokenAddress) {
    console.log("\nDeploying Translation contract...");
    
    // Deploy translation contract
    const TranslationContract = await hre.ethers.getContractFactory("TranslationContract");
    const translation = await TranslationContract.deploy(tokenAddress);
    await translation.deployed();
    
    console.log("Translation contract deployed to:", translation.address);
    await translation.deployTransaction.wait(5);
    
    return translation;
}

async function verifyContracts(token, translation) {
    if (process.env.ETHERSCAN_API_KEY) {
        console.log("\nVerifying contracts on Etherscan...");
        
        try {
            await hre.run("verify:verify", {
                address: token.address,
                constructorArguments: [],
            });
            console.log("Token contract verified");
            
            await hre.run("verify:verify", {
                address: translation.address,
                constructorArguments: [token.address],
            });
            console.log("Translation contract verified");
        } catch (error) {
            console.error("Error verifying contracts:", error);
        }
    }
}

async function initializeContracts(token, translation) {
    console.log("\nInitializing contracts...");
    
    // Initialize translation contract parameters
    const tx = await translation.initialize(
        hre.ethers.utils.parseEther(config.translation.minimumStake),
        config.translation.minimumValidations,
        config.translation.minimumScore,
        hre.ethers.utils.parseEther(config.translation.baseReward),
        config.translation.goodScoreMultiplier,
        config.translation.poorScoreMultiplier,
        config.translation.validatorRewardMultiplier
    );
    await tx.wait();
    
    // Get signers
    const [owner, treasury, ...otherSigners] = await hre.ethers.getSigners();
    
    // Transfer tokens to treasury
    const treasuryAmount = hre.ethers.utils.parseEther(config.translation.treasury.initialAmount);
    await token.transfer(treasury.address, treasuryAmount);
    
    // Transfer tokens to contract for rewards
    const contractRewards = hre.ethers.utils.parseEther(config.translation.rewards.contractAmount);
    await token.transfer(translation.address, contractRewards);
    
    // Grant contract permission to distribute rewards
    await token.grantRole(await token.REWARD_DISTRIBUTOR_ROLE(), translation.address);
    
    return { owner, treasury };
}

async function setupTestAccounts(token) {
    console.log("\nSetting up test accounts...");
    
    const [owner, treasury, ...otherSigners] = await hre.ethers.getSigners();
    const stakeAmount = hre.ethers.utils.parseEther(config.translation.testAccounts.stakeAmount);
    const testAccounts = otherSigners.slice(0, config.translation.testAccounts.count);
    
    for (const account of testAccounts) {
        // Transfer tokens
        await token.transfer(account.address, stakeAmount * 2);
        
        // Stake tokens
        await token.connect(account).stake(stakeAmount);
    }
}

async function saveDeploymentInfo(token, translation, network) {
    const deployments = {
        network: network.name,
        chainId: network.config.chainId,
        timestamp: Date.now(),
        token: {
            address: token.address,
            transactionHash: token.deployTransaction.hash,
        },
        translation: {
            address: translation.address,
            transactionHash: translation.deployTransaction.hash,
            tokenAddress: token.address,
        },
    };
    
    const deploymentsPath = path.join(__dirname, "..", "deployments.json");
    fs.writeFileSync(deploymentsPath, JSON.stringify(deployments, null, 2));
    console.log("\nDeployment info saved to deployments.json");
}

async function main() {
    try {
        const network = await hre.ethers.provider.getNetwork();
        console.log(`\nDeploying to network: ${network.name} (chainId: ${network.chainId})`);
        
        // Deploy contracts
        const token = await deployToken();
        const translation = await deployTranslation(token.address);
        
        // Verify contracts
        await verifyContracts(token, translation);
        
        // Initialize contracts
        const { owner, treasury } = await initializeContracts(token, translation);
        
        // Setup test accounts
        await setupTestAccounts(token);
        
        // Save deployment info
        await saveDeploymentInfo(token, translation, network);
        
        console.log("\nDeployment completed successfully!");
        console.log("Token address:", token.address);
        console.log("Translation contract address:", translation.address);
        console.log("Treasury address:", treasury.address);
        console.log("Owner address:", owner.address);
        
    } catch (error) {
        console.error("\nDeployment failed:", error);
        process.exit(1);
    }
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error(error);
        process.exit(1);
    }); 