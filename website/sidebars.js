// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  mainSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Getting Started',
      collapsed: false,
      items: [
        'getting-started/installation',
        'getting-started/quickstart',
      ],
    },
    {
      type: 'category',
      label: 'User Guide',
      collapsed: false,
      items: [
        'user-guide/automatic',
        'user-guide/manual',
        'user-guide/decontx',
        'user-guide/doublets',
        'user-guide/gene-het',
        'user-guide/iterative',
        'user-guide/downstream',
        'user-guide/visualization',
      ],
    },
    {
      type: 'category',
      label: 'API Reference',
      items: [
        'api/soup-channel',
        'api/io',
        'api/estimation',
        'api/correction',
        'api/metrics',
      ],
    },
    'results',
    'datasets',
    'contributing',
    'changelog',
  ],
};

export default sidebars;
