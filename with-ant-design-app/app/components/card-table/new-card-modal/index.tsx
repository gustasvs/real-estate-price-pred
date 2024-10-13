"use client";

import React, { useState } from 'react';
import { Button, Input, Modal } from 'antd';
import styles from './NewGroupModal.module.css'; // Import the CSS module

interface NewGroupModalProps {
  open: boolean;
  setOpen: (open: boolean) => void;
  addGroup: (groupName: string) => void;
}

const NewGroupModal: React.FC<NewGroupModalProps> = ({ open, setOpen, addGroup }) => {
  const [groupName, setGroupName] = useState('');

  const handleOk = () => {
    addGroup(groupName);
    setOpen(false);
    setGroupName('');  // Reset the input field after adding the group
  };

  const handleClose = () => {
    setOpen(false);
    setGroupName('');  // Reset the input field on close
  };

  return (
    <Modal
      title="Create New Group"
      open={open}
      onOk={handleOk}
      onCancel={handleClose}
      footer={[
        <Button key="back" onClick={handleClose} className={styles.cancelButton}>
          Cancel
        </Button>,
        <Button key="submit" type="primary" onClick={handleOk} className={styles.submitButton}>
          Add Group
        </Button>,
      ]}
      className={styles.modal}
    >
      <Input
        placeholder="Enter group name"
        value={groupName}
        onChange={(e) => setGroupName(e.target.value)}
        className={styles.input}
        maxLength={50}
      />
    </Modal>
  );
};

export default NewGroupModal;
