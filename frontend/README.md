# Real Estate Price Prediction App

This application uses Next.js along with Ant Design to provide a user interface for predicting real estate prices. The backend is powered by Prisma and PostgreSQL.

## Running Locally on Windows

Follow these steps to set up and run the application on a Windows environment:

### Prerequisites
1. **PostgreSQL**: Download and install PostgreSQL from [the official site](https://www.postgresql.org/download/windows/). During installation:
   - Set a password for the `postgres` user.
   - Optionally, install pgAdmin to manage your databases via a GUI.

### Setup
2. **Create a PostgreSQL Database**:
   - Open pgAdmin and create a new database named `mydb`.

3. **Environment Configuration**:
   - Copy the `.env.example` file to a new file called `.env`.
   - Update the `DATABASE_URL` in the `.env` file to reflect your PostgreSQL credentials. Example:
     ```
     DATABASE_URL="postgresql://postgres:your_password@localhost:5432/mydb?schema=public"
     ```

4. **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/real-estate-price-pred.git
    ```

5. **Navigate to the Project Directory**:
    ```bash
    cd real-estate-price-pred/with-ant-design-app
    ```

6. **Install Dependencies**:
    ```bash
    npm install
    ```

7. **Generate Prisma Client**:
    ```bash
    npx prisma generate
    ```

8. **Apply Prisma Migrations**:
    ```bash
    npx prisma migrate dev
    ```

9. **Run the Development Server**:
    ```bash
    npm run dev
    ```

10. **Visit the Application**:
    - Open your browser and go to `http://localhost:3000` to see the app in action.

## Running Locally on Linux
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
    <!-- npx prisma db push -->
    docker exec -it with-ant-design-app_app_1 npx prisma migrate dev
    ```
6. Run the development server:
    ```bash
    npm run dev
    ```
7. Open your browser and visit `http://localhost:3000` to see the app in action.


