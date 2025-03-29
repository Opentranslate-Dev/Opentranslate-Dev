module.exports = {
    hardhat: {
        chainId: 31337,
        accounts: {
            mnemonic: "test test test test test test test test test test test junk",
            path: "m/44'/60'/0'/0",
            initialIndex: 0,
            count: 20,
            accountsBalance: "10000000000000000000000",
        },
    },
    localhost: {
        url: "http://127.0.0.1:8545",
        chainId: 31337,
        accounts: {
            mnemonic: "test test test test test test test test test test test junk",
            path: "m/44'/60'/0'/0",
            initialIndex: 0,
            count: 20,
            accountsBalance: "10000000000000000000000",
        },
    },
    goerli: {
        url: process.env.GOERLI_URL || "",
        accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : [],
        chainId: 5,
        verify: {
            etherscan: {
                apiUrl: "https://api-goerli.etherscan.io",
                apiKey: process.env.ETHERSCAN_API_KEY,
            },
        },
    },
    sepolia: {
        url: process.env.SEPOLIA_URL || "",
        accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : [],
        chainId: 11155111,
        verify: {
            etherscan: {
                apiUrl: "https://api-sepolia.etherscan.io",
                apiKey: process.env.ETHERSCAN_API_KEY,
            },
        },
    },
    mainnet: {
        url: process.env.MAINNET_URL || "",
        accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : [],
        chainId: 1,
        verify: {
            etherscan: {
                apiUrl: "https://api.etherscan.io",
                apiKey: process.env.ETHERSCAN_API_KEY,
            },
        },
    },
}; 