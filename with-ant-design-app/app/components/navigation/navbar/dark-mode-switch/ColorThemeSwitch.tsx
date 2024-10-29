import { MoonFilled, SunFilled } from "@ant-design/icons";
import styles from "./ColorThemeSwitch.module.css";

const ColorThemeSwitch = ({
  currentTheme,
  setCurrentTheme,
}: {
  currentTheme: string;
  setCurrentTheme: (theme: string) => void;
}) => {
  const isDark = currentTheme === "dark";

  return (
    <div className={styles["color-mode-container"]}>
      <div
        className={styles["active-indicator"]}
        style={{
          transform: `translateX(${isDark ? "100%" : "0"})`,
        }}
      />
      <div
        className={`${styles["color-mode-button"]} ${
          styles["light-button"]
        } ${!isDark ? styles["active"] : ""}`}
        onClick={() => setCurrentTheme("light")}
      >
        <SunFilled />
      </div>
      <div
        className={`${styles["color-mode-button"]} ${
          styles["dark-button"]
        } ${isDark ? styles["active"] : ""}`}
        onClick={() => setCurrentTheme("dark")}
      >
        <MoonFilled />
      </div>
    </div>
  );
};

export default ColorThemeSwitch;
