const hre = require("hardhat");

async function main() {
    console.log("Deploying Translation contract...");
    
    // Load token address from deployments
    const deployments = require("../deployments.json");
    if (!deployments.token || !deployments.token.address) {
        throw new Error("Token contract not deployed. Please deploy token first.");
    }
    
    // Deploy translation contract
    const TranslationContract = await hre.ethers.getContractFactory("TranslationContract");
    const translation = await TranslationContract.deploy(deployments.token.address);
    await translation.deployed();
    
    console.log("Translation contract deployed to:", translation.address);
    
    // Wait for a few block confirmations
    await translation.deployTransaction.wait(5);
    
    // Verify contract on Etherscan
    if (process.env.ETHERSCAN_API_KEY) {
        console.log("Verifying contract on Etherscan...");
        try {
            await hre.run("verify:verify", {
                address: translation.address,
                constructorArguments: [deployments.token.address],
            });
            console.log("Contract verified on Etherscan");
        } catch (error) {
            console.error("Error verifying contract:", error);
        }
    }
    
    // Save deployment info
    deployments.translation = {
        address: translation.address,
        network: hre.network.name,
        timestamp: Date.now(),
        tokenAddress: deployments.token.address,
    };
    
    const fs = require("fs");
    fs.writeFileSync(
        "../deployments.json",
        JSON.stringify(deployments, null, 2)
    );
    
    console.log("Deployment info saved to deployments.json");
    
    // Initialize contract with initial parameters
    console.log("Initializing contract parameters...");
    const tx = await translation.initialize(
        ethers.utils.parseEther("1000"), // Minimum stake
        3, // Minimum validations
        70, // Minimum score for completion
        ethers.utils.parseEther("100"), // Base reward
        120, // Good score multiplier (120%)
        80, // Poor score multiplier (80%)
        2 // Validator reward multiplier
    );
    await tx.wait();
    console.log("Contract parameters initialized");
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error(error);
        process.exit(1);
    }); 