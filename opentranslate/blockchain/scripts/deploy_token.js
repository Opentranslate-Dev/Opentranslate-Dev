const hre = require("hardhat");

async function main() {
    console.log("Deploying PUMPFUN token...");
    
    // Deploy token
    const PUMPFUNToken = await hre.ethers.getContractFactory("PUMPFUNToken");
    const token = await PUMPFUNToken.deploy();
    await token.deployed();
    
    console.log("PUMPFUN token deployed to:", token.address);
    
    // Wait for a few block confirmations
    await token.deployTransaction.wait(5);
    
    // Verify contract on Etherscan
    if (process.env.ETHERSCAN_API_KEY) {
        console.log("Verifying contract on Etherscan...");
        try {
            await hre.run("verify:verify", {
                address: token.address,
                constructorArguments: [],
            });
            console.log("Contract verified on Etherscan");
        } catch (error) {
            console.error("Error verifying contract:", error);
        }
    }
    
    // Save deployment info
    const deployments = require("../deployments.json");
    deployments.token = {
        address: token.address,
        network: hre.network.name,
        timestamp: Date.now(),
    };
    
    const fs = require("fs");
    fs.writeFileSync(
        "../deployments.json",
        JSON.stringify(deployments, null, 2)
    );
    
    console.log("Deployment info saved to deployments.json");
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error(error);
        process.exit(1);
    }); 