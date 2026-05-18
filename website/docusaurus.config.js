// @ts-check
import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Upgraded-SoupX',
  tagline: 'Ambient RNA contamination removal for droplet-based single-cell RNA-seq',
  favicon: 'img/favicon.svg',

  url: 'https://isratijk.github.io',
  baseUrl: '/Upgraded-soupX/',

  organizationName: 'IsratIJK',
  projectName: 'Upgraded-soupX',

  onBrokenLinks: 'warn',

  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          editUrl: 'https://github.com/IsratIJK/Upgraded-soupX/edit/main/website/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      image: 'img/pipeline_diagram.png',
      colorMode: {
        defaultMode: 'light',
        respectPrefersColorScheme: true,
      },
      announcementBar: {
        id: 'v1_7_0',
        content: '🚀 <strong>v1.7.0</strong> released - docs site, benchmark results page, and full Python pipeline.',
        backgroundColor: '#0d9488',
        textColor: '#ffffff',
        isCloseable: true,
      },
      navbar: {
        title: 'Upgraded-SoupX',
        logo: {
          alt: 'Upgraded-SoupX Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'mainSidebar',
            position: 'left',
            label: 'Docs',
          },
          {
            to: '/docs/results',
            label: 'Results',
            position: 'left',
          },
          {
            to: '/docs/api/soup-channel',
            label: 'API',
            position: 'left',
          },
          {
            to: '/docs/changelog',
            label: 'Changelog',
            position: 'left',
          },
          {
            href: 'https://github.com/IsratIJK/Upgraded-soupX',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Docs',
            items: [
              {label: 'Installation', to: '/docs/getting-started/installation'},
              {label: 'Quick Start', to: '/docs/getting-started/quickstart'},
              {label: 'Automatic Workflow', to: '/docs/user-guide/automatic'},
              {label: 'DecontX', to: '/docs/user-guide/decontx'},
            ],
          },
          {
            title: 'API Reference',
            items: [
              {label: 'SoupChannel', to: '/docs/api/soup-channel'},
              {label: 'I/O', to: '/docs/api/io'},
              {label: 'Estimation', to: '/docs/api/estimation'},
              {label: 'Correction', to: '/docs/api/correction'},
              {label: 'Metrics', to: '/docs/api/metrics'},
            ],
          },
          {
            title: 'More',
            items: [
              {label: 'Benchmark Results', to: '/docs/results'},
              {label: 'Datasets', to: '/docs/datasets'},
              {label: 'Contributing', to: '/docs/contributing'},
              {label: 'Changelog', to: '/docs/changelog'},
              {
                label: 'GitHub',
                href: 'https://github.com/IsratIJK/Upgraded-soupX',
              },
            ],
          },
          {
            title: 'Author',
            items: [
              {
                label: 'Israt Jahan Khan',
                href: 'https://www.isratjahankhan.com',
              },
              {
                label: 'LinkedIn',
                href: 'https://www.linkedin.com/in/isratijk/',
              },
              {
                label: 'Google Scholar',
                href: 'https://scholar.google.com/citations?user=n4mCE9QAAAAJ&hl=en',
              },
              {
                label: 'Email',
                href: 'mailto:isratjahankhanijk@gmail.com',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Upgraded-SoupX. Developed by Israt Jahan Khan. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
        additionalLanguages: ['python', 'bash'],
      },
      tableOfContents: {
        minHeadingLevel: 2,
        maxHeadingLevel: 4,
      },
    }),
};

export default config;
