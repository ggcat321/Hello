# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 22:28:47 2023

@author: crazy
"""
list1 = [1,2,3,4]
list2 = [0,1,2]
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
        
class Solution(object):
    def mergeTwoLists(self, list1, list2):

        head = ListNode()
        current = head
        while list1 and list2:
            if list1.val < list2.val:
                current.next = list1
                list1 = list1.next
            else:
                current.next = list2
                list2 = list2.next

            current = current.next

        current.next = list1 or list2
        return head.next
    
aa = Solution()
aa.mergeTwoLists(list1,list2)
