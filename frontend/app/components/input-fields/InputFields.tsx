import classNames from "classnames";
import styles from "./InputFields.module.css";
import React from "react";
export type InputLabelProps = {
  labelText: string;
};
type OtherProps = {
  children: React.ReactElement;
  active: boolean;
  value?: string;
};
export default function InputLabel({
  active,
  labelText,
  children,
  value,
}: InputLabelProps & OtherProps) {
  return (
    <div style={{ display: "flex", flexFlow: "column-reverse" }}>
      {children}
      <span
        className={classNames(styles.label, {
          [styles.floatingLabel]: active || value,
        })}
      >
        {labelText}
      </span>
    </div>
  );
}
