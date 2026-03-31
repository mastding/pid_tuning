module.exports = {
  content: [
    './index.html',
    './*.html',
    './public/**/*.html',
    './js/**/*.js'
  ],
  theme: {
    extend: {}
  },
  plugins: [
    require('@tailwindcss/typography')
  ]
};
