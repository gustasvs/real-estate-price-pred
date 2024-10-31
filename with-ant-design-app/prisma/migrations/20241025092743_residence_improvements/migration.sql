/*
  Warnings:

  - Added the required column `address` to the `Residence` table without a default value. This is not possible if the table is not empty.
  - Added the required column `area` to the `Residence` table without a default value. This is not possible if the table is not empty.
  - Added the required column `bathroomCount` to the `Residence` table without a default value. This is not possible if the table is not empty.
  - Added the required column `bedroomCount` to the `Residence` table without a default value. This is not possible if the table is not empty.
  - Added the required column `parkingCount` to the `Residence` table without a default value. This is not possible if the table is not empty.
  - Added the required column `predictedPrice` to the `Residence` table without a default value. This is not possible if the table is not empty.
  - Added the required column `price` to the `Residence` table without a default value. This is not possible if the table is not empty.

*/
-- AlterTable
ALTER TABLE "Residence" ADD COLUMN     "address" TEXT NOT NULL,
ADD COLUMN     "area" DOUBLE PRECISION NOT NULL,
ADD COLUMN     "bathroomCount" INTEGER NOT NULL, g
ADD COLUMN     "bedroomCount" INTEGER NOT NULL,
ADD COLUMN     "parkingCount" INTEGER NOT NULL,
ADD COLUMN     "predictedPrice" DOUBLE PRECISION NOT NULL,
ADD COLUMN     "price" DOUBLE PRECISION NOT NULL;
