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

    return { token, translation, deployments };
}

async function upgradeToken() {
    console.log("\nUpgrading PUMPFUN token...");
    
    // Deploy new token implementation
    const PUMPFUNToken = await hre.ethers.getContractFactory("PUMPFUNToken");
    const newToken = await PUMPFUNToken.deploy();
    await newToken.deployed();
    
    console.log("New token implementation deployed to:", newToken.address);
    await newToken.deployTransaction.wait(5);
    
    return newToken;
}

async function upgradeTranslation(tokenAddress) {
    console.log("\nUpgrading Translation contract...");
    
    // Deploy new translation implementation
    const TranslationContract = await hre.ethers.getContractFactory("TranslationContract");
    const newTranslation = await TranslationContract.deploy(tokenAddress);
    await newTranslation.deployed();
    
    console.log("New translation implementation deployed to:", newTranslation.address);
    await newTranslation.deployTransaction.wait(5);
    
    return newTranslation;
}

async function verifyUpgrades(token, translation) {
    if (process.env.ETHERSCAN_API_KEY) {
        console.log("\nVerifying upgraded contracts on Etherscan...");
        
        try {
            await hre.run("verify:verify", {
                address: token.address,
                constructorArguments: [],
            });
            console.log("New token contract verified");
            
            await hre.run("verify:verify", {
                address: translation.address,
                constructorArguments: [token.address],
            });
            console.log("New translation contract verified");
        } catch (error) {
            console.error("Error verifying contracts:", error);
        }
    }
}

async function updateDeploymentInfo(token, translation, network) {
    const deployments = {
        network: network.name,
        chainId: network.config.chainId,
        timestamp: Date.now(),
        token: {
            address: token.address,
            transactionHash: token.deployTransaction.hash,
            previousAddress: deployments.token.address,
        },
        translation: {
            address: translation.address,
            transactionHash: translation.deployTransaction.hash,
            tokenAddress: token.address,
            previousAddress: deployments.translation.address,
        },
    };
    
    const deploymentsPath = path.join(__dirname, "..", "deployments.json");
    fs.writeFileSync(deploymentsPath, JSON.stringify(deployments, null, 2));
    console.log("\nDeployment info updated in deployments.json");
}

async function main() {
    try {
        const network = await hre.ethers.provider.getNetwork();
        console.log(`\nUpgrading on network: ${network.name} (chainId: ${network.chainId})`);
        
        // Get current contracts
        const { token: oldToken, translation: oldTranslation, deployments } = await getContracts();
        
        // Upgrade contracts
        const newToken = await upgradeToken();
        const newTranslation = await upgradeTranslation(newToken.address);
        
        // Verify contracts
        await verifyUpgrades(newToken, newTranslation);
        
        // Update deployment info
        await updateDeploymentInfo(newToken, newTranslation, network);
        
        console.log("\nUpgrade completed successfully!");
        console.log("New token address:", newToken.address);
        console.log("New translation contract address:", newTranslation.address);
        console.log("Previous token address:", deployments.token.address);
        console.log("Previous translation contract address:", deployments.translation.address);
        
    } catch (error) {
        console.error("\nUpgrade failed:", error);
        process.exit(1);
    }
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error(error);
        process.exit(1);
    }); 