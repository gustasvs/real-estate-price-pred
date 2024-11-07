"use client";

import React, { useState } from 'react';
import { Button, Input, Modal } from 'antd';
import styles from './NewGroupModal.module.css'; // Import the CSS module
import { StyledTextField } from '../../my-profile/my-profile-form/MyProfileForm';

interface NewGroupModalProps {
  open: boolean;
  setOpen: (open: boolean) => void;
  addGroup: (groupName: string) => void;
  onSubmit: (groupName: string) => void;
  isEditing?: boolean;
  groupName: string;
  setGroupName: (groupName: string) => void;
}

const NewGroupModal: React.FC<NewGroupModalProps> = ({ open, setOpen, addGroup, onSubmit, isEditing, groupName, setGroupName }) => {

  const handleOk = () => {
    onSubmit(groupName);
    setOpen(false);
    setGroupName('');  // Reset the input field after operation
  };
  

  const handleClose = () => {
    setOpen(false);
    setGroupName('');  // Reset the input field on close
  };

  return (
    <Modal
      title={isEditing ? "Labot grupu" : "Izveidot jaunu grupu"}
      open={open}
      onOk={handleOk}
      onCancel={handleClose}
      footer={
        <div
          className={styles.footer}
        >
        <Button key="back" onClick={handleClose} className={styles.cancelButton}>
          Atcelt
        </Button>
        <Button key="submit" type="primary" onClick={handleOk} className={styles.submitButton}>
          {isEditing ? "Labot" : "Pievienot"}
        </Button>
        </div>
      }
      className={styles.modal}
    >
      {/* <Input
        placeholder="Enter group name"
        value={groupName}
        onChange={(e) => setGroupName(e.target.value)}
        className={styles.input}
        maxLength={50}
      /> */}
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
    // maxLength={50}
  />
      </div>
    </Modal>
  );
};

export default NewGroupModal;
