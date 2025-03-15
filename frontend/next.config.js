/** @type {import('next').NextConfig} */
const nextConfig = {
  webpack: (config, { isServer }) => {
    // Only run this on the client-side builds
    if (!isServer) {
      // Import MiniCssExtractPlugin
      const MiniCssExtractPlugin = require('mini-css-extract-plugin');
      
      // Add the plugin to the webpack config
      config.plugins.push(
        new MiniCssExtractPlugin({
          filename: 'static/css/[name].[contenthash].css',
          chunkFilename: 'static/css/[name].[contenthash].css',
        })
      );
      
      // Configure the CSS loading rules if needed
      const cssRules = config.module.rules.find(
        (rule) => rule.oneOf && Array.isArray(rule.oneOf)
      )?.oneOf;
      
      if (cssRules) {
        // Update CSS rules to use MiniCssExtractPlugin instead of style-loader
        for (const rule of cssRules) {
          if (rule.test && rule.test.test && (rule.test.test('.css') || rule.test.test('.module.css'))) {
            if (rule.use && Array.isArray(rule.use)) {
              rule.use = rule.use.map((loader) => {
                if (loader === 'style-loader' || (loader.loader && loader.loader === 'style-loader')) {
                  return MiniCssExtractPlugin.loader;
                }
                return loader;
              });
            }
          }
        }
      }
    }
    
    return config;
  },
};

module.exports = nextConfig; 