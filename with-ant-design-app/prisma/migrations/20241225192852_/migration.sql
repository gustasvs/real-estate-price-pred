/*
  Warnings:

  - You are about to drop the column `bathroomCount` on the `Residence` table. All the data in the column will be lost.
  - You are about to drop the column `bedroomCount` on the `Residence` table. All the data in the column will be lost.
  - You are about to drop the column `parkingCount` on the `Residence` table. All the data in the column will be lost.
  - Added the required column `buildingFloors` to the `Residence` table without a default value. This is not possible if the table is not empty.
  - Added the required column `floor` to the `Residence` table without a default value. This is not possible if the table is not empty.
  - Added the required column `roomCount` to the `Residence` table without a default value. This is not possible if the table is not empty.

*/
-- AlterTable
ALTER TABLE "Residence" DROP COLUMN "bathroomCount",
DROP COLUMN "bedroomCount",
DROP COLUMN "parkingCount",
ADD COLUMN     "buildingFloors" INTEGER NOT NULL,
ADD COLUMN     "elevatorAvailable" BOOLEAN NOT NULL DEFAULT false,
ADD COLUMN     "floor" INTEGER NOT NULL,
ADD COLUMN     "parkingAvailable" BOOLEAN NOT NULL DEFAULT false,
ADD COLUMN     "roomCount" INTEGER NOT NULL;
