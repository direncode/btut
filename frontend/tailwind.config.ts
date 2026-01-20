import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // Apple-inspired color palette
        apple: {
          gray: {
            50: '#fafafa',
            100: '#f5f5f7',
            200: '#e8e8ed',
            300: '#d2d2d7',
            400: '#86868b',
            500: '#6e6e73',
            600: '#515154',
            700: '#424245',
            800: '#2d2d2d',
            900: '#1d1d1f',
          },
          blue: {
            DEFAULT: '#0071e3',
            light: '#147ce5',
            dark: '#0066cc',
          },
          green: {
            DEFAULT: '#34c759',
            light: '#30d158',
          },
          red: {
            DEFAULT: '#ff3b30',
            light: '#ff453a',
          },
          orange: {
            DEFAULT: '#ff9500',
            light: '#ff9f0a',
          },
          purple: {
            DEFAULT: '#af52de',
            light: '#bf5af2',
          },
          cyan: {
            DEFAULT: '#32ade6',
            light: '#64d2ff',
          },
        },
      },
      fontFamily: {
        sans: [
          '-apple-system',
          'BlinkMacSystemFont',
          'SF Pro Display',
          'SF Pro Text',
          'Helvetica Neue',
          'Helvetica',
          'Arial',
          'sans-serif',
        ],
        mono: [
          'SF Mono',
          'Monaco',
          'Inconsolata',
          'Fira Code',
          'monospace',
        ],
      },
      fontSize: {
        'display-xl': ['96px', { lineHeight: '1.05', letterSpacing: '-0.03em', fontWeight: '700' }],
        'display-lg': ['80px', { lineHeight: '1.05', letterSpacing: '-0.03em', fontWeight: '700' }],
        'display': ['64px', { lineHeight: '1.1', letterSpacing: '-0.02em', fontWeight: '600' }],
        'headline': ['48px', { lineHeight: '1.1', letterSpacing: '-0.02em', fontWeight: '600' }],
        'title-lg': ['32px', { lineHeight: '1.2', letterSpacing: '-0.01em', fontWeight: '600' }],
        'title': ['24px', { lineHeight: '1.3', letterSpacing: '-0.01em', fontWeight: '600' }],
        'body-lg': ['21px', { lineHeight: '1.5', fontWeight: '400' }],
        'body': ['17px', { lineHeight: '1.5', fontWeight: '400' }],
        'caption': ['14px', { lineHeight: '1.4', fontWeight: '400' }],
        'micro': ['12px', { lineHeight: '1.4', fontWeight: '500' }],
      },
      boxShadow: {
        'apple-sm': '0 1px 3px rgba(0, 0, 0, 0.08)',
        'apple': '0 4px 12px rgba(0, 0, 0, 0.08)',
        'apple-lg': '0 8px 30px rgba(0, 0, 0, 0.12)',
        'apple-xl': '0 20px 60px rgba(0, 0, 0, 0.15)',
        'glow': '0 0 40px rgba(0, 113, 227, 0.3)',
        'glow-green': '0 0 40px rgba(52, 199, 89, 0.3)',
        'glow-red': '0 0 40px rgba(255, 59, 48, 0.3)',
      },
      borderRadius: {
        'apple': '12px',
        'apple-lg': '20px',
        'apple-xl': '28px',
      },
      animation: {
        'fade-in': 'fadeIn 0.6s ease-out',
        'slide-up': 'slideUp 0.6s ease-out',
        'slide-down': 'slideDown 0.4s ease-out',
        'scale-in': 'scaleIn 0.4s ease-out',
        'pulse-slow': 'pulse 3s ease-in-out infinite',
        'float': 'float 6s ease-in-out infinite',
        'shimmer': 'shimmer 2s linear infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        slideDown: {
          '0%': { opacity: '0', transform: 'translateY(-10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        scaleIn: {
          '0%': { opacity: '0', transform: 'scale(0.95)' },
          '100%': { opacity: '1', transform: 'scale(1)' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-10px)' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
      },
      backdropBlur: {
        'apple': '20px',
      },
    },
  },
  plugins: [],
}

export default config
