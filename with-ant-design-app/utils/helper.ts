import bcrypt from "bcryptjs";
export function saltAndHashPassword(password: any) {
  const saltRounds = 10; // Adjust the cost factor according to your security requirements
  const salt = bcrypt.genSaltSync(saltRounds); // Synchronously generate a salt
  const hash = bcrypt.hashSync(password, salt); // Synchronously hash the password
  return hash; // Return the hash directly as a string
}

export function generateRandomHashedString() {
  return bcrypt.hashSync(
    Math.random().toString(36).substring(2, 15) +
      Math.random().toString(36).substring(2, 15),
    10
  );
}