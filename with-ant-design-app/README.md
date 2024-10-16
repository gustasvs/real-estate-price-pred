
# TO RUN LOCALLY
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/real-estate-price-pred.git
    ```
2. Navigate to the project directory:
    ```bash
    cd real-estate-price-pred/with-ant-design-app
    ```
3. Install the dependencies:
    ```bash
    npm install
    ```
4. Generate Prisma client:
    ```bash
    npx prisma generate
    ```
5. Push the Prisma schema to the database:
    ```bash
    npx prisma db push
    ```
6. Run the development server:
    ```bash
    npm run dev
    ```
7. Open your browser and visit `http://localhost:3000` to see the app in action.






# Ant Design example

This example shows how to use Next.js along with [Ant Design of React](https://ant.design). This is intended to show the integration of this UI toolkit with the Framework.

## Deploy your own

Deploy the example using [Vercel](https://vercel.com?utm_source=github&utm_medium=readme&utm_campaign=next-example) or preview live with [StackBlitz](https://stackblitz.com/github/vercel/next.js/tree/canary/examples/with-ant-design)

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/vercel/next.js/tree/canary/examples/with-ant-design&project-name=with-ant-design&repository-name=with-ant-design)

## How to use

Execute [`create-next-app`](https://github.com/vercel/next.js/tree/canary/packages/create-next-app) with [npm](https://docs.npmjs.com/cli/init), [Yarn](https://yarnpkg.com/lang/en/docs/cli/create/), or [pnpm](https://pnpm.io) to bootstrap the example:

```bash
npx create-next-app --example with-ant-design with-ant-design-app
```

```bash
yarn create next-app --example with-ant-design with-ant-design-app
```

```bash
pnpm create next-app --example with-ant-design with-ant-design-app
```

Deploy it to the cloud with [Vercel](https://vercel.com/new?utm_source=github&utm_medium=readme&utm_campaign=next-example) ([Documentation](https://nextjs.org/docs/deployment)).
