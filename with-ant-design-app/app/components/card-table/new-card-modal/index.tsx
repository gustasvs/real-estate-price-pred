"use client";

import React, { useState } from "react";
import { Button, Input, message, Modal, Popconfirm } from "antd";
import styles from "./NewGroupModal.module.css";
import { StyledTextField } from "../../my-profile/my-profile-form/MyProfileForm";
import { CloseOutlined, ExclamationCircleOutlined } from "@ant-design/icons";
import { DeleteOutlineOutlined } from "@mui/icons-material";

interface NewGroupModalProps {
  open: boolean;
  setOpen: (open: boolean) => void;
  addGroup: (groupName: string) => void;
  deleteGroup: (id: string) => void;
  onSubmit: (groupName: string) => void;
  isEditing?: boolean;
  groupName: string;
  groupId: string;
  setGroupName: (groupName: string) => void;
}

const NewGroupModal: React.FC<NewGroupModalProps> = ({
  open,
  setOpen,
  addGroup,
  deleteGroup,
  onSubmit,
  isEditing,
  groupName,
  groupId,
  setGroupName,
}) => {
  const handleOk = () => {
    onSubmit(groupName);
    setOpen(false);
    setGroupName(""); // Reset the input field after operation
  };

  const handleClose = () => {
    setOpen(false);
    setGroupName(""); // Reset the input field on close
  };

  const handleDeleteGroup = (id: string) => {
    deleteGroup(id);
    setOpen(false);
    setGroupName(""); // Reset the input field after operation
    message.success("Grupa veiksmīgi dzēsta!");
  }

  return (
    <Modal
      title={
        isEditing ? "Rediģēt grupu" : "Izveidot jaunu grupu"
      }
      open={open}
      onOk={handleOk}
      onCancel={handleClose}
      width={600}
      closeIcon={<CloseOutlined className={styles.closeIcon} />}
      footer={
        <div className={styles.footer}>
          {isEditing && (
            <Popconfirm
              title="Vai tiešām vēlaties dzēst šo grupu?"
              onConfirm={() => handleDeleteGroup(groupId)}
              okText="Jā"
              cancelText="Nē"
              okButtonProps={{ style: { padding: "0.5em 1em", borderRadius: "1em", marginTop: ".5em" } }}
              color="var(--background-darkest)"
              cancelButtonProps={{ style: { padding: "0.5em 1em", borderRadius: "1em", marginTop: ".5em" } }}
              icon={<ExclamationCircleOutlined style={{ color: "var(--background-light-main)" }} />}
            >
              {/* <Tooltip title="Dzēst grupu"> */}
              <Button
                key="delete"
                danger
                className={styles.deleteButton}
              >
                <DeleteOutlineOutlined />
              </Button>
              {/* </Tooltip> */}
            </Popconfirm>
          )}
          <Button
            key="back"
            onClick={handleClose}
            className={styles.cancelButton}
          >
            Atcelt
          </Button>
          <Button
            key="submit"
            type="primary"
            onClick={handleOk}
            className={styles.submitButton}
          >
            {isEditing ? "Saglabāt izmaiņas" : "Pievienot"}
          </Button>
        </div>
      }
      className={styles.modal}
    >
      <div
        style={{
          marginTop: "2em",
        }}
      >
        <StyledTextField

          id="outlined-basic"
          label="Grupas nosaukums"
          variant="outlined"
          value={groupName}
          onChange={(e) => setGroupName(e.target.value)}
          className={styles.input}
          slotProps={{ htmlInput: { maxLength: 50 } }}
          helperText={`${groupName.length}/50`}
        />
      </div>
    </Modal>
  );
};

export default NewGroupModal;
