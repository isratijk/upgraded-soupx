import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import CodeBlock from '@theme/CodeBlock';
import styles from './index.module.css';

const FEATURES = [
  {
    icon: '/Upgraded-soupX/img/icons/icon-dna.svg',
    title: 'Full Python Port',
    description:
      'Complete reimplementation of the R SoupX package. No R dependency - runs natively in the Python / scipy.sparse ecosystem.',
  },
  {
    icon: '/Upgraded-soupX/img/icons/icon-cell.svg',
    title: 'DecontX Per-Cell Decontamination',
    description:
      'Two-component Dirichlet-Multinomial EM with LDA topics. Estimates per-cell contamination theta instead of a single global rho.',
  },
  {
    icon: '/Upgraded-soupX/img/icons/icon-target.svg',
    title: 'Doublet-Aware Estimation',
    description:
      'Scrublet-style doublet scoring masks contaminated doublets before rho estimation, producing cleaner per-cluster contamination rates.',
  },
  {
    icon: '/Upgraded-soupX/img/icons/icon-loop.svg',
    title: 'Iterative Refinement',
    description:
      'Runs auto_est_cont - adjust_counts - soup profile update until convergence. Achieves the best cluster preservation across all datasets.',
  },
  {
    icon: '/Upgraded-soupX/img/icons/icon-gene.svg',
    title: 'Gene Heterogeneity Correction',
    description:
      'Reweights the soup profile to amplify truly ambient genes. Reduces spurious DE genes by up to 98% on barnyard data.',
  },
  {
    icon: '/Upgraded-soupX/img/icons/icon-metrics.svg',
    title: '8 Quantitative Metrics',
    description:
      'Cross-species reduction, marker fold change, cluster ARI, batch entropy, HBB analysis, silhouette, spurious DE, marker enrichment.',
  },
];

const STATS = [
  {number: '5', label: 'Decontamination Pipelines'},
  {number: '8', label: 'Benchmark Metrics'},
  {number: '5', label: 'Benchmark Datasets'},
  {number: '98%', label: 'Spurious DE Reduction (HGMM)'},
];

const QUICK_CODE = `from SoupX import load_10x, set_clusters, auto_est_cont, adjust_counts

# Load CellRanger output (v2 or v3, auto-detected)
sc = load_10x('path/to/cellranger/outs/')

# Attach cluster labels from Seurat / Scanpy
sc = set_clusters(sc, cluster_labels)

# Estimate contamination fraction rho automatically
sc = auto_est_cont(sc)
print(f"Contamination: {sc.meta_data['rho'].mean():.1%}")

# Produce corrected count matrix
corrected = adjust_counts(sc)`;

function FeatureCard({icon, title, description}) {
  return (
    <div className={clsx('col col--4', styles.featureCol)}>
      <div className={styles.featureCard}>
        <img src={icon} className={styles.featureIcon} alt="" aria-hidden="true" />
        <Heading as="h3" className={styles.featureTitle}>{title}</Heading>
        <p className={styles.featureDesc}>{description}</p>
      </div>
    </div>
  );
}

function StatItem({number, label}) {
  return (
    <div className={clsx('col col--3', styles.statCol)}>
      <span className={styles.statNumber}>{number}</span>
      <span className={styles.statLabel}>{label}</span>
    </div>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title="Upgraded-SoupX"
      description="Ambient RNA contamination removal for droplet-based single-cell RNA-seq. Full Python port of SoupX with DecontX, doublet-aware estimation, iterative refinement, and 8 benchmark metrics.">

      {/* Hero */}
      <header className={clsx('hero hero--primary', styles.heroBanner)}>
        <div className="container">
          <Heading as="h1" className="hero__title">
            Upgraded-SoupX
          </Heading>
          <p className="hero__subtitle">
            Ambient RNA contamination removal for droplet-based single-cell RNA-seq.<br />
            Full Python port - DecontX - Doublet-aware - Iterative refinement - 8 metrics
          </p>
          <p className={styles.heroAuthor}>
            by{' '}
            <a href="https://www.isratjahankhan.com" style={{color: 'inherit', textDecoration: 'underline'}}>
              Israt Jahan Khan
            </a>
          </p>
          <div className={styles.heroCtas}>
            <Link className={styles.btnPrimary} to="/docs/getting-started/installation">
              Get Started
            </Link>
            <Link className={styles.btnGhost} to="/docs/results">
              View Results
            </Link>
            <Link className={styles.btnGhost} href="https://github.com/IsratIJK/Upgraded-soupX">
              GitHub
            </Link>
          </div>
        </div>
      </header>

      <main>
        {/* Stats strip */}
        <section className={styles.statsStrip}>
          <div className="container">
            <div className="row">
              {STATS.map((s, i) => <StatItem key={i} {...s} />)}
            </div>
          </div>
        </section>

        {/* Features */}
        <section className={styles.features}>
          <div className="container">
            <div className={styles.featuresSectionHeader}>
              <Heading as="h2">Beyond the R Baseline</Heading>
              <p>Everything the original SoupX R package provides, plus six new decontamination modes and a full benchmark framework.</p>
            </div>
            <div className="row">
              {FEATURES.map((f, i) => <FeatureCard key={i} {...f} />)}
            </div>
          </div>
        </section>

        {/* Quick example */}
        <section className={styles.quickExampleSection}>
          <div className="container">
            <div className={clsx('row', styles.quickExampleRow)}>
              <div className={clsx('col col--5', styles.quickExampleText)}>
                <Heading as="h2">30-second example</Heading>
                <p>
                  Load CellRanger output, add cluster labels, estimate contamination
                  automatically, and correct counts - all in four function calls.
                </p>
                <p>
                  Works with <strong>v2 and v3 CellRanger</strong> layouts, MEX and HDF5
                  formats, and integrates with any clustering tool (Seurat, Scanpy, etc.).
                </p>
                <div className={styles.quickLinks}>
                  <Link to="/docs/getting-started/quickstart">Full Quick Start -&gt;</Link>
                  <Link to="/docs/user-guide/automatic">Automatic Workflow -&gt;</Link>
                  <Link to="/docs/user-guide/decontx">DecontX -&gt;</Link>
                </div>
              </div>
              <div className="col col--7">
                <CodeBlock language="python" title="30-second example">
                  {QUICK_CODE}
                </CodeBlock>
              </div>
            </div>
          </div>
        </section>

        {/* Pipeline comparison table */}
        <section className={styles.pipelineSection}>
          <div className="container">
            <Heading as="h2" className="text--center">Pipeline Comparison</Heading>
            <p className="text--center" style={{marginBottom: '2rem'}}>
              Choose the pipeline that best fits your data and biological question.
            </p>
            <table className={styles.pipelineTable}>
              <thead>
                <tr>
                  <th>Pipeline</th>
                  <th>Rho Type</th>
                  <th>Best For</th>
                  <th>Cluster Preservation</th>
                  <th>Speed</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><code>auto_est_cont</code></td>
                  <td>Global / per-cluster</td>
                  <td>Most datasets</td>
                  <td>Good</td>
                  <td>Fast</td>
                </tr>
                <tr>
                  <td><code>iterative_auto_est_cont</code></td>
                  <td>Global / per-cluster</td>
                  <td>Overlapping soup + cells</td>
                  <td>Best (highest ARI)</td>
                  <td>Moderate</td>
                </tr>
                <tr>
                  <td><code>run_decontx</code></td>
                  <td>Per-cell</td>
                  <td>Barnyard, heterogeneous tissues</td>
                  <td>Good</td>
                  <td>Slow</td>
                </tr>
                <tr>
                  <td><code>run_decontx_genehet</code></td>
                  <td>Per-cell</td>
                  <td>Blood contamination, high spurious DE</td>
                  <td>Moderate</td>
                  <td>Slow</td>
                </tr>
                <tr>
                  <td><code>auto_est_cont_doublet_aware</code></td>
                  <td>Global / per-cluster</td>
                  <td>High doublet rate experiments</td>
                  <td>Good</td>
                  <td>Fast</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Installation */}
        <section className={styles.installSection}>
          <div className="container">
            <div className={clsx('row', styles.installRow)}>
              <div className="col col--6">
                <Heading as="h2">Installation</Heading>
                <CodeBlock language="bash">
                  {`# Core package
pip install -e .

# With downstream analysis (PCA, UMAP, Leiden):
pip install -e ".[downstream]"`}
                </CodeBlock>
                <p>Requires Python &ge; 3.9.</p>
                <Link to="/docs/getting-started/installation">Full installation guide -&gt;</Link>
              </div>
              <div className="col col--6">
                <Heading as="h2">Citation</Heading>
                <p style={{fontSize: '0.85rem', marginBottom: '0.4rem', fontWeight: 600}}>Cite this software:</p>
                <blockquote className={styles.citation}>
                  <p>
                    Khan, I.J. (2026). <em>Upgraded-SoupX: A Python port and extension of SoupX
                    for ambient RNA decontamination in single-cell RNA-seq.</em>{' '}
                    GitHub. <a href="https://github.com/IsratIJK/Upgraded-soupX">github.com/IsratIJK/Upgraded-soupX</a>
                  </p>
                </blockquote>
                <p style={{fontSize: '0.85rem', marginBottom: '0.4rem', fontWeight: 600, marginTop: '1rem'}}>Also cite the original algorithms:</p>
                <blockquote className={styles.citation}>
                  <p>
                    Young, M.D. &amp; Behjati, S. (2020). SoupX removes ambient RNA
                    contamination from droplet-based single-cell RNA sequencing data.{' '}
                    <em>GigaScience</em>, 9(12), giaa151.
                  </p>
                </blockquote>
                <blockquote className={styles.citation}>
                  <p>
                    Yang, S. et al. (2020). Decontamination of ambient RNA in single-cell
                    RNA-seq with DecontX. <em>Genome Biology</em>, 21, 57.
                  </p>
                </blockquote>
                <div className={styles.authorNote}>
                  <p style={{marginBottom: '0.4rem'}}>
                    Developed by{' '}
                    <strong>
                      <a href="https://www.isratjahankhan.com">Israt Jahan Khan</a>
                    </strong>.
                  </p>
                  <p style={{margin: 0, fontSize: '0.85rem'}}>
                    <a href="https://www.linkedin.com/in/isratijk/">LinkedIn</a>
                    {' · '}
                    <a href="https://scholar.google.com/citations?user=n4mCE9QAAAAJ&hl=en">Google Scholar</a>
                    {' · '}
                    <a href="mailto:isratjahankhanijk@gmail.com">isratjahankhanijk@gmail.com</a>
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}
