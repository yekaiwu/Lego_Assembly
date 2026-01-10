import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        lego: {
          red: '#D01012',
          yellow: '#FFC425',
          blue: '#0055BF',
          green: '#00852B',
        },
      },
    },
  },
  plugins: [],
}
export default config




