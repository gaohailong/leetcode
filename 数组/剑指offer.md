- O:原始
- P:改进
- T:思考
---
###  二维数组中的查找
>  在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
- O:
```
    public boolean Find(int target, int [][] array) {
        for(int [] cells : array){
            for(int cell : cells){
                if (target == cell){
                    return true;
                }
            }
        }
        return false;
    }
```
- P1:
```
	public boolean Find(int target, int [][] array) {
        int rows = array.length;
        int cols = array[0].length;
        int i = rows - 1, j = 0;//左下角元素坐标
        while (i >= 0 && j < cols) {//使其不超出数组范围
            if (target < array[i][j]) {
                i--;//查找的元素较少，往上找
            } else if (target > array[i][j]) {
                j++;//查找元素较大，往右找
            } else {
                return true;//找到
            }
        }
        return false;
    }
```
---
### 替换空格
> 请实现一个函数，将一个字符串中的每个空格替换成“%20”。例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。
- O：
```
    public String replaceSpace(StringBuffer str) {
        if (str == null) {
            return null;
        }
        for (int i = 0; i < str.length(); i++) {
            char chr = str.charAt(i);
            if (chr == ' ') {
                str.delete(i, i + 1);
                str.insert(i, "%20");
                i = i + 2;
            }
        }
        return str.toString();
    }
```
---
### 从尾到头打印链表
>输入一个链表，按链表从尾到头的顺序返回一个ArrayList。
- O(递归):
```
	ArrayList<Integer> arrayList = new ArrayList<>();
    public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
        if (listNode == null){
            return arrayList;
        }
        this.printListFromTailToHead(listNode.next);
        arrayList.add(listNode.val);
        return arrayList;
    }
```
---
### 重建二叉树
>输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。
- O:
```
    public TreeNode reConstructBinaryTree(int[] pre, int[] in) {
        if (pre.length == 0 || in.length == 0) {
            return null;
        }
        TreeNode treeNode = new TreeNode(pre[0]);
        for (int i = 0; i < in.length; i++) {
            if (pre[0] == in[i]) {
                // copyOfRange 左闭右开
                treeNode.left = reConstructBinaryTree(Arrays.copyOfRange(pre, 1, i + 1), Arrays.copyOfRange(in, 0, i));
                treeNode.right = reConstructBinaryTree(Arrays.copyOfRange(pre, i + 1, pre.length), Arrays.copyOfRange(in, i + 1, in.length));
            }
        }
        return treeNode;
    }
```
---
### 用两个栈实现队列
> 用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型。
- O:
```
Stack<Integer> stack1 = new Stack<Integer>();
    Stack<Integer> stack2 = new Stack<Integer>();
    
     public void push(int node) {
        stack1.push(node);
    }

    public int pop() {
        if (!stack2.isEmpty()){
            return stack2.pop();
        }
        while (!stack1.isEmpty()){
            stack2.push(stack1.pop());
        }
        return stack2.pop();
    }
```
---
### 旋转数组的最小数字
> 把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。
输入一个非递减排序的数组的一个旋转，输出旋转数组的最小元素。
例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。
NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。
- O（封啸自创解法，真牛皮）:
```
	public int minNumberInRotateArray(int[] array) {
        if (array.length == 0) {
            return 0;
        }
        int len = array.length;
        int low = 0;
        int high = len - 1;
        while (low < high) {
            int mid = low + high >> 1;
            if (array[mid] <= array[len - 1]) {//如果左侧小于右侧，
                high = mid;
            } else {
                low = mid + 1;
            }
        }
        return array[low];
    }
```
- P1:
```
	public int minNumberInRotateArray(int [] array) {
        for(int i = 0;i<array.length-1;i++){
            if(array[i]>array[i+1]){
                return array[i+1];
            }
        }
        return array[0];
    }
```
- T:
1. 不一定非要考虑题中的旋转。通过求出最小亦是求出答案。
通过观察可知。如果array[mid]<=array[high] 可知，如果成立，那么符合二分查找。数据最小的一定在左边。其他情况。说明不成立。那么范围一定得从高位中找。
2. 
正数：r = 20 << 2;结果：r = 80

负数：r = -20 << 2;结果：r = -80

正数：r = 20 >> 2;结果：r = 5

负数：r = -20 >> 2;结果：r = -5

二分查找：
```
int BinarySearch(int array[], int n, int value)
{
    int left = 0;
    int right = n - 1;
    //如果这里是int right = n 的话，那么下面有两处地方需要修改，以保证一一对应：
    //1、下面循环的条件则是while(left < right)
    //2、循环内当 array[middle] > value 的时候，right = mid

    while (left <= right)  //循环条件，适时而变
    {
        int middle = left + ((right - left) >> 1);  //防止溢出，移位也更高效。同时，每次循环都需要更新。
        if (array[middle] > value)
            right = middle - 1;  //right赋值，适时而变
        else if (array[middle] < value)
            left = middle + 1;
        else
            return middle;
        //可能会有读者认为刚开始时就要判断相等，但毕竟数组中不相等的情况更多
        //如果每次循环都判断一下是否相等，将耗费时间
    }
    return -1;
}
```
---
### 斐波那契数列
> 大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项（从0开始，第0项为0）。
n<=39
- O:
```
	public int Fibonacci(int n) {
        if(n == 0){
            return 0;
        }else if(n == 1 ||n == 2){
            return 1;
        }else{
            return Fibonacci(n-1)+Fibonacci(n-2);
        }
    }
```
- P:
```
 	public int Fibonacci(int n) {
        if(n <= 1){
            return n;
        }
        int count = 0;
        int tempOne = 0;
        int tempTwo = 1;
        for(int i = 2;i<= n;i++){
            count = tempOne + tempTwo;
            tempOne = tempTwo;
            tempTwo = count;
        }
        return count;
    }
```
- T:
1.  斐波那契数列
            1                                x = 1
f(x) =   1       x = 2
           f(x - 1)  + f(x - 2)        x >= 3
---
### 跳台阶
> 一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。
- O:
```
	public int JumpFloor(int target) {
        if(target<=0){
            return 0;
        }
        if(target == 1){
            return 1;
        } 
        if(target == 2){
            return 2;
        } 
        return JumpFloor(target-1)+JumpFloor(target-2);
    }
```
- P:
```
public int JumpFloor(int target) {
        if(target<=0){
            return 0;
        }
        if(target == 1){
            return 1;
        } 
        if(target == 2){
            return 2;
        } 
        int count = 0;
        int resOne = 1;
        int resTwo = 2;
        for(int i = 2; i < target;i++){
            count = resOne + resTwo;
            resOne = resTwo;
            resTwo = count;
        }
        return count;
    }
```
---
### 变态跳台阶
> 一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。
- O:
```
	public int JumpFloorII(int target) {
        if(target<=0){
            return 0;
        }
        if(target == 1){
            return 1;
        } 
        if(target == 2){
            return 2;
        } 
        int count = 2;
        for(int i = 2;i<target;i++){
            count = 2* count;
        }
        return count;
    }
```
---
### 矩形覆盖
> 我们可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？
- O:
```
 public int RectCover(int target) {
        if(target == 0){
            return 0;
        }
        if(target == 1){
            return 1;
        }
        if(target == 2){
            return 2;
        }
        return RectCover(target-1)+RectCover(target-2);
    }
```
---
### <font color="#dd0000">二进制中1的个数</font>
>输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。 
- O:
```
 	public int NumberOf1(int n) {
        int count = 0;
        while(n!= 0){
            count++;
            n = n & (n - 1);
         }
        return count;
    }
```
- T:
```
&按位与的运算规则是将两边的数转换为二进制位，然后运算最终值，运算规则即(两个为真才为真)1&1=1 , 1&0=0 , 0&1=0 , 0&0=0
```
---
### 数值的整数次方

> 给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。保证base和exponent不同时为0

- O:
```
public double Power(double base, int exponent) {
        return Math.pow(base, exponent);
    }
```
- P:
```
	public double Power(double base, int exponent) {
        int temp = exponent>0? exponent : -exponent;
        double result = 1;
        for(int i = 0 ;i < temp; i++){
            result *= base;
        }
        return exponent > 0 ?  result : 1 / result;
    }
```
---
### 调整数组顺序使奇数位于偶数前面
> 输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。
- O:
```
	public void reOrderArray(int [] array) {
        for(int i = 0;i< array.length;i++){
            for( int j = 0;j<array.length-i-1;j++ ){
                if(array[j]%2==0&&array[j+1]%2==1){
                    int temp = array[j];
                    array[j] = array[j+1];
                    array[j+1] = temp;
                }
            }
        }
    }
```
---
### 链表中倒数第k个结点
> 输入一个链表，输出该链表中倒数第k个结点。
- O:
```
	public ListNode FindKthToTail(ListNode head, int k) {
        ListNode pre = head;
        ListNode post = head;
        int prePos = 0;
        int postPos = 0;
        while (post != null) {
            post = post.next;
            postPos++;
            if (postPos - prePos > k) {
                prePos++;
                pre = pre.next;
            }
        }
        return postPos < k ? null : pre;
    }
```
---
### 反转链表
> 输入一个链表，反转链表后，输出新链表的表头。
- O:
```
 public ListNode ReverseList(ListNode head) {
	// 思路：逐个断裂，并且转向
        if (head == null) {
            return null;
        }
        ListNode pre = null;
        ListNode next = null;
        while (head != null) {
            next = head.next; //  把head 后面的元素给next
            head.next = pre;//把后面元素的指向转为前面

            //进行后移操作
            pre = head; //  把当前节点给 pre，相当于pre后移
            head = next;// head 也后移
        }
        return pre;
   }
```
---
### 合并两个排序的链表
> 输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。
- O:
```
public ListNode Merge(ListNode list1,ListNode list2) {
        ListNode node = new ListNode(-1);
        node.next=null;
        ListNode root = node;
        while(list1!=null&&list2!=null){
            if(list1.val>list2.val){
                node.next = list2;
                node=list2;
                list2 = list2.next;
            }else{
                node.next = list1;
                node=list1;
                list1 = list1.next;
            }
        }
        if(list1!=null){
            node.next = list1;
        }
        if(list2!=null){
            node.next = list2;
        }
        return root.next;
    }
```
---
### 树的子结构
> 输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）
- O:
```
	public boolean isSubTree(TreeNode root1, TreeNode root2) {
        if (root2 == null) { // root2 为空时，说明已经比较完毕
            return true;
        }
        if (root1 == null) { // root1 为空时，说明root1长度不够
            return false;
        }
        if (root1.val == root2.val) {
            return isSubTree(root1.left, root2.left) && isSubTree(root1.right, root2.right);
        } else {
            return false;
        }
    }

    public boolean HasSubtree(TreeNode root1, TreeNode root2) {
        if (root1 == null || root2 == null) {
            return false;
        }
        return isSubTree(root1, root2) || HasSubtree(root1.left, root2) || HasSubtree(root1.right, root2);
    }

```
---
### 二叉树的镜像
> 操作给定的二叉树，将其变换为源二叉树的镜像。
```
二叉树的镜像定义：源二叉树 
    	    8
    	   /  \
    	  6   10
    	 / \  / \
    	5  7 9 11
    	镜像二叉树
    	    8
    	   /  \
    	  10   6
    	 / \  / \
    	11 9 7  5
```
- O:
```
	public void Mirror(TreeNode root) {
         if (root!=null){
            TreeNode temp = root.left;
            root.left = root.right;
            root.right = temp;
            Mirror(root.left);
            Mirror(root.right);
        }
    }
```
- P1(非递归):
```
	public void Mirror(TreeNode root) {
        if (root == null) {
            return;
        }
        Stack<TreeNode> stack = new Stack<TreeNode>();
        stack.push(root);
        while (!stack.isEmpty()) {
            TreeNode treeNode = stack.pop();
            if (treeNode.left != null || treeNode.right != null) {
                TreeNode temp = treeNode.left;
                treeNode.left = treeNode.right;
                treeNode.right = temp;
            }
            if (treeNode.left != null) {
                stack.push(treeNode.left);
            }
            if (treeNode.right != null) {
                stack.push(treeNode.right);
            }
        }
    }
```
---
### 顺时针打印矩阵
> 输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，例如，如果输入如下4 X 4矩阵： 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.

- O:
```
	public ArrayList<Integer> printMatrix(int[][] matrix) {
        ArrayList<Integer> list = new ArrayList<>();
        if (matrix == null || matrix.length == 0) {
            return list;
        }

        int top = 0;
        int bottom = matrix.length - 1;
        int left = 0;
        int right = matrix[0].length - 1;

        while (true) {
            for (int i = left; i <= right; i++) {
                list.add(matrix[top][i]);
            }
            top++;
            if (top > bottom) {
                break;
            }
            for (int i = top; i <= bottom; i++) {
                list.add(matrix[i][right]);
            }
            right--;
            if (right < left) {
                break;
            }
            for (int i = right; i >= left; i--) {
                list.add(matrix[bottom][i]);
            }
            bottom--;
            if (bottom < top) {
                break;
            }
            for (int i = bottom; i >= top; i--) {
                list.add(matrix[i][left]);
            }
            left++;
            if (left>right) {
                break;
            }
        }
        return list;
    }
```
- T:
```
四个坐标朝中心点逼近
```
---
### 包含min函数的栈
> 定义栈的数据结构，请在该类型中实现一个能够得到栈中所含最小元素的min函数（时间复杂度应为O（1））。
- O:
```
 Stack<Integer> statck = new Stack<Integer>();
    
    public void push(int node) {
        statck.push(node);
    }
    
    public void pop() {
        statck.pop();
    }
    
    public int top() {
        return statck.peek();
    }
    
    public int min() {
        int min = statck.peek();
        int tmp = 0;
        Iterator<Integer> iterator = statck.iterator();
        while (iterator.hasNext()){
            tmp = iterator.next();
            if (min>tmp){
                min = tmp;
            }
        }
        return min;
    }
```
---
### 栈的压入、弹出序列
> 输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如序列1,2,3,4,5是某栈的压入顺序，序列4,5,3,2,1是该压栈序列对应的一个弹出序列，但4,3,5,1,2就不可能是该压栈序列的弹出序列。（注意：这两个序列的长度是相等的）
- O: 
```
	public boolean IsPopOrder(int [] pushA,int [] popA) {
      if(pushA.length == 0||popA.length == 0){
          return false;
      }
      Stack<Integer> stack = new Stack<Integer>();
        int index= 0;
      for(int i = 0;i<pushA.length;i++){
          stack.push(pushA[i]);
          // 添加的时候判断是否存在与当前出栈顺序相等的，如果最后为空，肯定是正确的
          while(!stack.empty()&&stack.peek()==popA[index]){
              stack.pop();
              index++;
          }
      }
        return stack.empty();
    }
```
---
### 从上往下打印二叉树
> 从上往下打印出二叉树的每个节点，同层节点从左至右打印。
- O:
```
public ArrayList<Integer> PrintFromTopToBottom(TreeNode root) {
        ArrayList<Integer> arrayList = new ArrayList<Integer>();
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        if (root == null) {
            return arrayList;
        }
        queue.add(root);
        while (!queue.isEmpty()) {
            TreeNode treeNode = queue.poll();
            arrayList.add(treeNode.val);

            if (treeNode.left != null) {
                queue.add(treeNode.left);
            }
            if (treeNode.right != null) {
                queue.add(treeNode.right);
            }
        }
        return arrayList;
    }
```
---
### <font color="red">二叉搜索树的后序遍历序列</font>
> 输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。
- O:
```
	public boolean VerifySquenceOfBST(int [] sequence) {
        if(sequence==null||sequence.length==0){
            return false;
        }
        return helper(sequence,0,sequence.length-1);
    }
    public boolean helper(int [] sequence,int start,int root){
        if(start>=root){
            return true;
        }
        int key = sequence[root];
        // 找到左右子树的分界点
        int i;
        for( i=start;i<root;i++){
            if(sequence[i]>key){
                break;
            }
        }
         //在右子树中判断是否含有小于root的值，如果有返回false
        for(int j =i;j<root;j++){
            if(sequence[j]<key){
                return false;
            }
        }
        return helper(sequence,0,i-1)&&helper(sequence,i,root-1);
    }
```
---
### <font color="red">二叉树中和为某一值的路径</font>
> 输入一颗二叉树的跟节点和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。(注意: 在返回值的list中，数组长度大的数组靠前)

- O:
```
 public ArrayList<ArrayList<Integer>> FindPath(TreeNode root, int target) {
        if (root == null) {
            return listAll;
        }
        list.add(root.val);
        target -= root.val;
        if (target == 0 && root.left == null && root.right == null) {
            listAll.add(new ArrayList<Integer>(list));
        }
        FindPath(root.left, target);
        FindPath(root.right, target);
        list.remove(list.size() - 1); // 如果走到这，说明这个条件不符合，剩下的没必要往下找了。直接剪枝。
        return listAll;
    }

```
---
### <font color="red">复杂链表的复制</font>
> 输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针指向任意一个节点），返回结果为复制后复杂链表的head。（注意，输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）
- O:
```
	public RandomListNode Clone(RandomListNode pHead){
        if(pHead == null){
            return null;
        }
        // 复制节点
        RandomListNode currentNode = pHead;
        while(currentNode!=null){
            RandomListNode cloneNode = new RandomListNode(currentNode.label);
            RandomListNode nextNode = currentNode.next;
            currentNode.next = cloneNode;
            cloneNode.next = nextNode;
            currentNode = nextNode;
        }
        // 复制随机节点
        currentNode = pHead;
        while(currentNode !=null){
            currentNode.next.random = currentNode.random == null?null:currentNode.random.next;
            currentNode = currentNode.next.next;
        }
        // 拆分
        currentNode = pHead;
        RandomListNode pCloneHead = pHead.next;
        while(currentNode != null) {
            RandomListNode cloneNode = currentNode.next;
            currentNode.next = cloneNode.next;
            cloneNode.next = cloneNode.next==null?null:cloneNode.next.next;
            currentNode = currentNode.next;
        }
         
        return pCloneHead;
    }
```
---
### 二叉搜索树与双向链表
> 输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向。
- O:
```
public TreeNode Convert(TreeNode pRootOfTree) {
        if(pRootOfTree == null){
            return null;
        }
        Stack<TreeNode> stack = new Stack<TreeNode>();
        //中序遍历
        TreeNode node = pRootOfTree;
        TreeNode root = null;
        TreeNode pre = null;
        boolean isFirst = true;
        while(node!=null|| !stack.isEmpty()){
            while(node !=null){
                stack.push(node);
                node = node.left;
            }
            node = stack.pop();
            //判断是否是第一次
            if(isFirst){
                root = node;
                pre = node;
                isFirst = false;
            }else{
                pre.right = node;
                node.left = pre;
                pre = node;
            }
            node = node.right;
        }
        return root;
    }
```
---
### *字符串的排列
> 输入一个字符串,按字典序打印出该字符串中字符的所有排列。例如输入字符串abc,则打印出由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。
- O:
```
 //固定第一个字符，递归取得首位后面的各种字符串组合；再将第一个字符与后面每一个字符交换，同样递归获得其字符串组合；
    // 每次递归都是到最后一位时结束，递归的循环过程，就是从每个子串的第二个字符开始依次与第一个字符交换，然后继续处理子串。
    public ArrayList<String> Permutation(String str) {
        ArrayList<String> list = new ArrayList<String>();
        if (str == null || str.length() == 0) {
            return list;
        }
        char[] strs = str.toCharArray();
        cal(strs, 0, list);
        Collections.sort(list);
        return list;
    }

    public void cal(char[] strs, int i, ArrayList<String> list) {
        if (i == strs.length - 1) {
            if (!list.contains(new String(strs))) {
                list.add(new String(strs));
            }
        } else {
            for (int j = i; j < strs.length; j++) {
                swap(strs, i, j); // 交换
                cal(strs, i + 1, list);
                swap(strs, i, j); // 还原
            }
        }
    }

    private void swap(char[] chars, int i, int j) {
        if (i != j) {
            char temp = chars[i];
            chars[i] = chars[j];
            chars[j] = temp;
        }
    }
```
---
### 数组中出现次数超过一半的数字
> 数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。
- O:
```
	public int MoreThanHalfNum_Solution(int [] array) {
        if(array.length==1){
            return array[0];
        }
        int len = array.length / 2;
        int[] counts = new int[array.length]; 
        for (int i = 0; i < array.length; i++) {
            int value = counts[array[i]];
            value++;
            if (value > len) {
                return array[i];
            }
            counts[array[i]]++;
        }
        return 0;
    }
```
- P1:
 通过先排序，排序之后如果数据大于一半，那么排序之后除以2，中间的数据肯定是这个数，不然他就无法大于一半。
```
public int MoreThanHalfNum_Solution(int [] array) {
        Arrays.sort(array);
        int len = array.length/2;
        int center = array[len]; // 如果大于一半，这个数据肯定在中间,但是不确定这个数是不是
        int count =0;
        for(int temp:array){
            if(temp==center){
                count++;
            }
        }
        if(count>len){
            return center;
        }
        return 0;
    }
```
---
### 最小的K个数
> 输入n个整数，找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4,。
- O:
```
	public ArrayList<Integer> GetLeastNumbers_Solution(int [] input, int k) {
        ArrayList<Integer> list = new ArrayList<Integer>();
        if(input.length <k){
            return list;
        }
        Arrays.sort(input);
        for(int i = 0;i< k;i++){
            list.add(input[i]);
        }
        return list;
    }
```
- P1:
```
 public ArrayList<Integer> GetLeastNumbers_Solution(int[] input, int k) {
        ArrayList<Integer> arr = new ArrayList<Integer>();
        if (input.length < k) {
            return arr;
        }
        quick(input, 0, input.length - 1);
        for (int i = 0; i < k; i++) {
            arr.add(input[i]);
        }
        return arr;
    }
    
    public void quick(int[] input, int start, int end) {
        int i = start;
        int j = end;
        if (i > j) { //放在k之前，防止下标越界
            return;
        }
        int base = input[start];
        while (i < j) {
            while (i < j && base <= input[j]) {
                j--;
            }
            while (i < j && base >= input[i]) {
                i++;
            }
            if (i < j) {
                int t = input[i];
                input[i] = input[j];
                input[j] = t;
            }
        }
        // 交换数据
        int temp = input[i];
        input[i] = base;
        input[start] = temp;

        quick(input, start, i - 1);
        quick(input, i + 1, end);
    }
```
---
### *连续子数组的最大和
> HZ偶尔会拿些专业问题来忽悠那些非计算机专业的同学。今天测试组开完会后,他又发话了:在古老的一维模式识别中,常常需要计算连续子向量的最大和,当向量全为正数的时候,问题很好解决。但是,如果向量中包含负数,是否应该包含某个负数,并期望旁边的正数会弥补它呢？例如:{6,-3,-2,7,-15,1,2,2},连续子向量的最大和为8(从第0个开始,到第3个为止)。给一个数组，返回它的最大连续子序列的和，你会不会被他忽悠住？(子向量的长度至少是1)
- O:
```
  public int FindGreatestSumOfSubArray(int[] array) {
        //两个值，分别指向值判断是否需要加
        int count = 0;
        int ans = array[0];
        for(int temp : array){
            if(count >0){
                count += temp;
            }else{
                count = temp;
            }
            ans = Math.max(ans,count);
        }
        return ans;
    }

原生
public class Solution {
    public int FindGreatestSumOfSubArray(int[] array) {
        int len = array.length;
        int[] dp = new int[len];
        int max = array[0];
        dp[0] = array[0];
        for(int i = 1; i < len; i++){
            int newmax = dp[i - 1] + array[i];
            if(newmax > array[i])
                dp[i] = newmax;
            else
                dp[i] = array[i];
            if(dp[i] > max)
                max = dp[i];
            
        }
        return max;
    }
}

```
---
### 整数中1出现的次数（从1到n整数中1出现的次数）
> 求出1~13的整数中1出现的次数,并算出100~1300的整数中1出现的次数？为此他特别数了一下1~13中包含1的数字有1、10、11、12、13因此共出现6次,但是对于后面问题他就没辙了。ACMer希望你们帮帮他,并把问题更加普遍化,可以很快的求出任意非负整数区间中1出现的次数（从1 到 n 中1出现的次数）
```
	public int test21(int n) {
        int count = 0;
        while (n > 0) {
            String temp = String.valueOf(n);
            char[] chars = temp.toCharArray();
            for (char c : chars) {
                if (c == '1') {
                    count++;
                }
            }
            n--;
        }
        return count;
    }

```
---
### 把数组排成最小的数
> 输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。
```
public String PrintMinNumber(int[] numbers) {
        ArrayList<Integer> list = new ArrayList<Integer>();
        for (int i = 0; i < numbers.length; i++) {
            list.add(numbers[i]);
        }
        Collections.sort(list, new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                String c1 = o1 + "" + o2;
                String c2 = o2 + "" + o1;
                return c1.compareTo(c2);
            }
        });
        StringBuffer sb = new StringBuffer();
        for (int i : list) {
            sb.append(i);
        }
        return sb.toString();
    }
```
---
### 丑数
> 把只包含质因子2、3和5的数称作丑数（Ugly Number）。例如6、8都是丑数，但14不是，因为它包含质因子7。 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。
- O :
```
 // 选定第一个丑数1，根据丑数的定义，可知以后的丑数必然是在1的基础上乘以2，乘以3，乘以5，
    // 因此可以得出三个丑数，从中选择最小的一个添加到list列表中，之后若list中的丑数与得出的三个丑数中的一个或两个相等，
    // 将对应的下标后移
    public int GetUglyNumber_Solution(int index) {
        if (index == 0) {
            return 0;
        }
        ArrayList<Integer> list = new ArrayList<>();
        list.add(1);
        int t2 = 0, t3 = 0, t5 = 0;
        while (list.size() < index) {
            int m2 = list.get(t2) * 2;
            int m3 = list.get(t3) * 3;
            int m5 = list.get(t5) * 5;
            int min = Math.min(m2, Math.min(m3, m5));
            list.add(min);
            if (min == m2) {
                t2++;
            }
            if (min == m3) {
                t3++;
            }
            if (min == m5) {
                t5++;
            }
        }
        return list.get(list.size() - 1);
    }
```
---
### 第一个只出现一次的字符
```
在一个字符串(0<=字符串长度<=10000，全部由字母组成)中找到第一个只出现一次的字符,并返回它的位置, 如果没有则返回 -1（需要区分大小写）.
```
- O :
```
 	public static int FirstNotRepeatingChar(String str) {
        char[] chars = str.toCharArray();
        HashMap<Character, Integer> map = new HashMap<>();
        for (int i = 0; i < chars.length; i++) {
            if (map.containsKey(chars[i])) {
                int count = map.get(chars[i]);
                map.put(chars[i],count+1);
            } else {
                map.put(chars[i], 1);
            }
        }
        if (map.isEmpty()) {
            return -1;
        } else {
           for (int i = 0;i< chars.length;i++){
               if (map.containsKey(chars[i])&&map.get(chars[i])==1){
                   return i;
               }
           }
        }
        return -1;
    }
```
---
### 数组中的逆序对
> 在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组,求出这个数组中的逆序对的总数P。并将P对1000000007取模的结果输出。 即输出P%1000000007
- O:
```
public int InversePairs(int[] array) {
        int count = 0;
        for (int i = 0; i < array.length; i++) {
            for (int j = i + 1; j < array.length; j++) {
                if (array[i] > array[j]) {
                    count++;
                }
            }
        }
        return count % 1000000007;
    }
```
- P:
```

```
---
### 两个链表的第一个公共结点
> 输入两个链表，找出它们的第一个公共结点。
- O:
```
public ListNode FindFirstCommonNode(ListNode pHead1, ListNode pHead2) {
        if(pHead1 == null ||pHead2 ==null ){
            return null;
        }
        ListNode p1 =pHead1;
        while (p1 != null) {
            ListNode p2 =pHead2;
            while (p2 != null) {
                if (p1 == p2) {
                    return p1;
                } else {
                    p2 = p2.next;
                }
            }
            p1 = p1.next;
        }
        return null;
    }
```
- P1:
```
	public ListNode FindFirstCommonNode(ListNode pHead1, ListNode pHead2) {
        if (pHead1 == null || pHead2 == null) {
            return null;
        }
        Stack<ListNode> stack1 = new Stack<>();
        Stack<ListNode> stack2 = new Stack<>();
        while (pHead1 != null) {
            stack1.push(pHead1);
            pHead1 = pHead1.next;
        }
        while (pHead2 != null) {
            stack2.push(pHead2);
            pHead2 = pHead2.next;
        }
        ListNode temp = null;
        while (!stack1.isEmpty() && !stack2.isEmpty()) {
            ListNode temp1 = stack1.pop();
            ListNode temp2 = stack2.pop();
            if (temp1.val == temp2.val) {
                temp = temp1;
            }
        }
        return temp;
    }
```
---
### 数字在排序数组中出现的次数
> 统计一个数字在排序数组中出现的次数。
- O:
```
public int GetNumberOfK(int [] array , int k) {
       int count = 0;
       for( int i = 0;i<array.length;i++){
           if(array[i] == k){
               count ++;
           }
       }
        return count;
    }
```
- P1:
```
public int GetNumberOfK(int [] array , int k) {
      return binarySearch(array,k+0.5)-binarySearch(array,k-0.5);
    }
    public int binarySearch(int [] array , double k){
        int s = 0,e=array.length-1;
        while(s<=e){
            int mid = (s+e)/2;
            if(array[mid]>k){
                e= mid-1;
            }else if(array[mid] <k){
                s=mid+1;
            }
        }
        return s;
    }
```
---
### 二叉树的深度
> 输入一棵二叉树，求该树的深度。从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，最长路径的长度为树的深度。
- O:
```
public int TreeDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int left = TreeDepth(root.left);
        int right = TreeDepth(root.right);
        return left > right ? left+1 : right+1;
    }
```
- P1:非递归遍历
```
 public int TreeDepth(TreeNode root) {
        if(root==null){
            return 0;
        }
       int count = 0;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while(!queue.isEmpty()){
            count++;
           int cnt = queue.size();
            for(int i =0;i<cnt;i++){
               TreeNode temp = queue.poll();
                if(temp.left!=null){
                    queue.add(temp.left);
                }
                if(temp.right!=null){
                    queue.add(temp.right);
                }
            }
        }
        return count;
    }
```
---
### 平衡二叉树
> 输入一棵二叉树，判断该二叉树是否是平衡二叉树。
- O:
```
   public boolean IsBalanced_Solution(TreeNode root) {
        if (root == null) {
            return true;
        }
        int a = Math.abs(calDeep(root.left) - calDeep(root.right));
        if (a > 1) {
            return false;
        }
        return true;
    }

    public int calDeep(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int left = calDeep(root.left);
        int right = calDeep(root.right);
        return left > right ? left + 1 : right + 1;
    }
```
---
### 数组中只出现一次的数字
> 一个整型数组里除了两个数字之外，其他的数字都出现了两次。请写程序找出这两个只出现一次的数字。
>
> - O:  
```
public void FindNumsAppearOnce(int[] array, int num1[], int num2[]) {
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < array.length; i++) {
            if (map.containsKey(array[i])) {
                int count = map.get(array[i]);
                map.put(array[i], count + 1);
            } else {
                map.put(array[i], 1);
            }
        }
        ArrayList<Integer> arr = new ArrayList<>();
        for (int temp : map.keySet()) {
            if (map.get(temp) == 1) {
                arr.add(temp);
            }
        }
        num1[0] = arr.get(0);
        num2[0] = arr.get(1);
    }
```
---
### 和为S的连续正数序列
> 小明很喜欢数学,有一天他在做数学作业时,要求计算出9~16的和,他马上就写出了正确答案是100。但是他并不满足于此,他在想究竟有多少种连续的正数序列的和为100(至少包括两个数)。没多久,他就得到另一组连续正数和为100的序列:18,19,20,21,22。现在把问题交给你,你能不能也很快的找出所有和为S的连续正数序列? Good Luck!
- O:
```
    public ArrayList<ArrayList<Integer>> FindContinuousSequence(int sum) {
        ArrayList<ArrayList<Integer>> arr = new ArrayList<>();
        int low = 1, high = 2;
        while (low < high) {
            int num = (low + high) * (high - low + 1) / 2;
            if (num == sum) {
                ArrayList<Integer> list = new ArrayList<Integer>();
                for (int i = low; i <= high; i++) {
                    list.add(i);
                }
                arr.add(list);
                low++;
            }
            if (num < sum) {
                //如果当前窗口内的值之和小于sum，那么右边窗口右移一下
                high++;
            }
            if (num > sum) {
                //如果当前窗口内的值之和大于sum，那么左边窗口右移一下
                low++;
            }
        }
        return arr;
    }
```
---
### 和为S的两个数字
> 输入一个递增排序的数组和一个数字S，在数组中查找两个数，使得他们的和正好是S，如果有多对数字的和等于S，输出两个数的乘积最小的。(对应每个测试案例，输出两个数，小的先输出。)
- O:
```
 	public ArrayList<Integer> FindNumbersWithSum(int [] array,int sum) {
         ArrayList<Integer> arr = new ArrayList<Integer>();
        int low = 0;
        int high = array.length - 1;
        while (low < high) {
            if ((array[low] + array[high]) == sum) {
                arr.add(array[low]);
                arr.add(array[high]);
                return arr;
            } else if ((array[low] + array[high]) > sum) {
                high--;
            } else {
                low++;
            }
        }
        return arr;
    }
```
---
### 左旋转字符串
> 汇编语言中有一种移位指令叫做循环左移（ROL），现在有个简单的任务，就是用字符串模拟这个指令的运算结果。对于一个给定的字符序列S，请你把其循环左移K位后的序列输出。例如，字符序列S=”abcXYZdef”,要求输出循环左移3位后的结果，即“XYZdefabc”。是不是很简单？OK，搞定它！
- O:
```
public String LeftRotateString(String str, int n) {
        if (str == null|| str.length() == 0){
            return "";
        }
        if(n>str.length()){
            return "";
        }
        String left = str.substring(0,n);
        String right = str.substring(n,str.length());
        String fin = right +left;

        return fin;
    }
```
 - P1:
```
	public String LeftRotateString(String str, int n) {
        char[] chars = str.toCharArray();
        if(str == null || str.length() == 0)
            return "";
        if(n > str.length())
            n = n % str.length();

        reverse(chars, 0, n - 1);
        reverse(chars, n, chars.length - 1);
        reverse(chars, 0, chars.length - 1);
        return new String(chars);
    }

    public void reverse(char[] chars, int start, int end) {
        while (start < end) {
            char temp = chars[start];
            chars[start] = chars[end];
            chars[end] = temp;
            start++;
            end--;
        }
    }
```
---
### 翻转单词顺序列
> 牛客最近来了一个新员工Fish，每天早晨总是会拿着一本英文杂志，写些句子在本子上。同事Cat对Fish写的内容颇感兴趣，有一天他向Fish借来翻看，但却读不懂它的意思。例如，“student. a am I”。后来才意识到，这家伙原来把句子单词的顺序翻转了，正确的句子应该是“I am a student.”。Cat对一一的翻转这些单词顺序可不在行，你能帮助他么？
- O :
```
 public String ReverseSentence(String str) {
          if(str == null){ return null;}
         if(str.trim().equals("")){
            return str;
        }
        String[] temp = str.split(" ");
        StringBuffer sb = new StringBuffer();
        for (int i = temp.length - 1; i >= 0; i--) {
            sb.append(temp[i]);
            if (i != 0) {
                sb.append(" ");
            }
        }
        return sb.toString();
    }
```
---
### *扑克牌顺子
> LL今天心情特别好,因为他去买了一副扑克牌,发现里面居然有2个大王,2个小王(一副牌原本是54张^_^)...他随机从中抽出了5张牌,想测测自己的手气,看看能不能抽到顺子,如果抽到的话,他决定去买体育彩票,嘿嘿！！“红心A,黑桃3,小王,大王,方片5”,“Oh My God!”不是顺子.....LL不高兴了,他想了想,决定大\小 王可以看成任何数字,并且A看作1,J为11,Q为12,K为13。上面的5张牌就可以变成“1,2,3,4,5”(大小王分别看作2和4),“So Lucky!”。LL决定去买体育彩票啦。 现在,要求你使用这幅牌模拟上面的过程,然后告诉我们LL的运气如何， 如果牌能组成顺子就输出true，否则就输出false。为了方便起见,你可以认为大小王是0。
- O:
```
public boolean isContinuous(int [] numbers) {
        if(numbers.length == 0){
            return false;
        }
        int[] temp = new int[14];
        int max = -1;
        int min = 14;
        for (int i = 0; i < numbers.length; i++) {
            temp[numbers[i]]++;
            if(numbers[i] == 0){
                continue;
            }
            if (temp[numbers[i]] > 1) { // 重复肯定不是
                return false;
            }
            if (numbers[i] > max){
                max = numbers[i];
            }
            if (numbers[i] < min){
                min = numbers[i];
            }
        }
        if(max-min<5){
            return true;
        }else {
            return false;
        }
    }
```
---
### 孩子们的游戏(圆圈中最后剩下的数)
> 每年六一儿童节,牛客都会准备一些小礼物去看望孤儿院的小朋友,今年亦是如此。HF作为牛客的资深元老,自然也准备了一些小游戏。其中,有个游戏是这样的:首先,让小朋友们围成一个大圈。然后,他随机指定一个数m,让编号为0的小朋友开始报数。每次喊到m-1的那个小朋友要出列唱首歌,然后可以在礼品箱中任意的挑选礼物,并且不再回到圈中,从他的下一个小朋友开始,继续0...m-1报数....这样下去....直到剩下最后一个小朋友,可以不用表演,并且拿到牛客名贵的“名侦探柯南”典藏版(名额有限哦!!^_^)。请你试着想下,哪个小朋友会得到这份礼品呢？(注：小朋友的编号是从0到n-1)
 - O:
```
	public int LastRemaining_Solution(int n, int m) {
        LinkedList<Integer> linkedList = new LinkedList<>();
        if (n == 0 || m == 0) {
            return -1;
        }
        int curr = 0;
        for (int i = 0; i < n; i++) {
            linkedList.add(i);
        }
        while (linkedList.size() > 1) {
            curr = (curr + m - 1) % (linkedList.size());
            linkedList.remove(curr);
        }
        return linkedList.get(0);
    }
```
---
### *求1+2+3+...+n
>求1+2+3+...+n，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。 
- O:
```
 public int Sum_Solution(int n) {
        int sum = n;
//当n==0时，(n>0)&&((sum+=Sum_Solution(n-1))>0)只执行前面的判断，为false，然后直接返回0；
        boolean ans = (n>0) && ((sum += Sum_Solution(n -1))>0);
        return sum;
    }
```
---
### *i不用加减乘除做加法
> 写一个函数，求两个整数之和，要求在函数体内不得使用+、-、*、/四则运算符号。
- O:
```
 	public int Add(int num1,int num2) {
        while (num2!=0) {
            int temp = num1^num2;
            num2 = (num1&num2)<<1;
            num1 = temp;
        }
        return num1;
    }
```
- T:
```
&（按位与）
&按位与的运算规则是将两边的数转换为二进制位，然后运算最终值，运算规则即(两个为真才为真)1&1=1 , 1&0=0 , 0&1=0 , 0&0=0
7的二进制位是0000 0111，那就是111 & 101等于101，也就是0000 0101，故值为5

|（按位或）
|按位或和&按位与计算方式都是转换二进制再计算，不同的是运算规则(一个为真即为真)1|0 = 1 , 1|1 = 1 , 0|0 = 0 , 0|1 = 1
6的二进制位0000 0110 , 2的二进制位0000 0010 , 110|010为110，最终值0000 0110，故6|2等于6

^（异或运算符）
^异或运算符顾名思义，异就是不同，其运算规则为1^0 = 1 , 1^1 = 0 , 0^1 = 1 , 0^0 = 0
5的二进制位是0000 0101 ， 9的二进制位是0000 1001，也就是0101 ^ 1001,结果为1100 , 00001100的十进制位是12

<<（左移运算符）
5<<2的意思为5的二进制位往左挪两位，右边补0，5的二进制位是0000 0101 ， 就是把有效值101往左挪两位就是0001 0100 ，正数左边第一位补0，负数补1，等于乘于2的n次方，十进制位是20

>>（右移运算符）
凡位运算符都是把值先转换成二进制再进行后续的处理，5的二进制位是0000 0101，右移两位就是把101左移后为0000 0001，正数左边第一位补0，负数补1，等于除于2的n次方，结果为1

~（取反运算符）
取反就是1为0,0为1,5的二进制位是0000 0101，取反后为1111 1010，值为-6

>>>（无符号右移运算符）
无符号右移运算符和右移运算符的主要区别在于负数的计算，因为无符号右移是高位补0，移多少位补多少个0。
15的二进制位是0000 1111 ， 右移2位0000 0011，结果为3

```
---
### 把字符串转换成整数
> 将一个字符串转换成一个整数，要求不能使用字符串转换整数的库函数。 数值为0或者字符串不是一个合法的数值则返回0
- O:
```
	public int StrToInt(String str) {
        if(str.trim().equals("")){
            return 0;
        }
        char[] nums = str.toCharArray();
        
        int zf = 1;
        int i =0;
        if(nums[0]=='-'){
            zf = -1;
            i=1;
        }
        if(nums[0]=='+'){
            zf = 1;
            i=1;
        }
        int value =0;
        int overValue = 0;
        int digit = 0;
        for(;i<nums.length;i++){
            digit=nums[i]-'0';
            overValue = zf*value-Integer.MAX_VALUE/10+(((zf + 1) / 2 + digit > 8) ? 1 : 0);
            if(digit<0||digit>9){
                return 0;
            }
            if(overValue>0){
                return 0;
            }
            value =value*10+digit*zf;
        }
        return value;
    }
```
---
### 数组中重复的数字
> 在一个长度为n的数组里的所有数字都在0到n-1的范围内。 数组中某些数字是重复的，但不知道有几个数字是重复的。也不知道每个数字重复几次。请找出数组中任意一个重复的数字。 例如，如果输入长度为7的数组{2,3,1,0,2,5,3}，那么对应的输出是第一个重复的数字2。
- O:
```
public boolean duplicate(int numbers[],int length,int [] duplication) {
        if (numbers == null ||numbers.length ==0){
            return false;
        }
        Arrays.sort(numbers);
        int temp = numbers[0];
        boolean isDup = false;
        for (int i = 1; i < numbers.length; i++) {
            if (temp == numbers[i]) {
                isDup = true;
                duplication[0] = temp;
                break;
            } else {
                temp = numbers[i];
            }
        }
        return isDup;
    }
```
- P1:
```
 public boolean duplicate(int numbers[],int length,int [] duplication) {
        Set<Integer> set = new HashSet<>();
        for(int i =0 ;i<length;i++){
            if(set.contains(numbers[i])){
                duplication[0] = numbers[i];
                return true;
            }else{
                set.add(numbers[i]);
            }
        }
        return false;
    }
```
<font color="#dd0000">
> 给定一个数组A[0,1,...,n-1],请构建一个数组B[0,1,...,n-1],其中B中的元素B[i]=A[0]*A[1]*...*A[i-1]*A[i+1]*...*A[n-1]。不能使用除法。</font>
- O:
```
int length = A.length;
        int[] B = new int[length];
        if (length != 0) {
            B[0] = 1;
            //下三角
            for (int i = 1; i < length; i++) {
                B[i] = B[i - 1] * A[i - 1];
            }
            //上三角
            int temp = 1;
            for (int j = length - 2; j >= 0; j--) {
                temp *= A[j + 1];
                B[j] *= temp;
            }
        }
        return B;
```
---
### 正则表达式匹配
> 请实现一个函数用来匹配包括'.'和'*'的正则表达式。模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（包含0次）。 在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但是与"aa.a"和"ab*a"均不匹配
- O:
```
public boolean matchStr(char[] str, int i, char[] pattern, int j) {
 
     // 边界
     if (i == str.length && j == pattern.length) { // 字符串和模式串都为空
         return true;
     } else if (j == pattern.length) { // 模式串为空
         return false;
     }
 
     boolean flag = false;
     boolean next = (j + 1 < pattern.length && pattern[j + 1] == '*'); // 模式串下一个字符是'*'
     if (next) {
         if (i < str.length && (pattern[j] == '.' || str[i] == pattern[j])) { // 要保证i<str.length，否则越界
             return matchStr(str, i, pattern, j + 2) || matchStr(str, i + 1, pattern, j);
         } else {
             return matchStr(str, i, pattern, j + 2);
         }
     } else {
         if (i < str.length && (pattern[j] == '.' || str[i] == pattern[j])) {
             return matchStr(str, i + 1, pattern, j + 1);
         } else {
             return false;
         }
     }
 }
 
 public boolean match(char[] str, char[] pattern) {
     return matchStr(str, 0, pattern, 0);
 }
```
---
### 表示数值的字符串
> 请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。例如，字符串"+100","5e2","-123","3.1416"和"-1E-16"都表示数值。 但是"12e","1a3.14","1.2.3","+-5"和"12e+4.3"都不是。
- O:
```

```
---
### 字符流中第一个不重复的字符
```
题目描述
请实现一个函数用来找出字符流中第一个只出现一次的字符。例如，当从字符流中只读出前两个字符"go"时，第一个只出现一次的字符是"g"。当从该字符流中读出前六个字符“google"时，第一个只出现一次的字符是"l"。
输出描述:
如果当前字符流没有存在出现一次的字符，返回#字符。
```
- O:
```
 
```
---
### 链表中环的入口结点
> 给一个链表，若其中包含环，请找出该链表的环的入口结点，否则，输出null。
- O:
```
public ListNode EntryNodeOfLoop(ListNode pHead) {
        ListNode fast = pHead;
        ListNode low = pHead;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            low = low.next;
            if (fast == low) {
                break;
            }
        }
        if (fast == null || fast.next == null) {
            return null;
        }
        low = pHead;
        while (fast != low) {
            fast = fast.next;
            low = low.next;
        }
        return low;
    }
```
- T:
```
当两个点相遇时，fast肯定是low的两倍。这样第二次low从新开始时候，fast依然从相遇点出发，这样他们再次相交时候就是入口点。
```
---
### 删除链表中重复的结点
> 在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留，返回链表头指针。 例如，链表1->2->3->3->4->4->5 处理后为 1->2->5
- O:
```
	public ListNode deleteDuplication(ListNode pHead) {
        if (pHead == null || pHead.next == null) {
            return pHead;
        }
        ListNode Head = new ListNode(0);
        Head.next = pHead;
        ListNode pre = Head;
        ListNode last = Head.next;
        while (last != null) {
            if (last.next != null && last.val == last.next.val) {
                // 找到最后的一个相同节点
                while (last.next != null && last.val == last.next.val) {
                    last = last.next;
                }
                pre.next = last.next;
                last = last.next;
            } else {
                pre = pre.next;
                last = last.next;
            }
        }
        return Head.next;
    }
```
---
### 二叉树的下一个结点
> 给定一个二叉树和其中的一个结点，请找出中序遍历顺序的下一个结点并且返回。注意，树中的结点不仅包含左右子结点，同时包含指向父结点的指针。
- O:
```
  public TreeLinkNode GetNext(TreeLinkNode pNode){
        if(pNode==null){
            return null;
        }
        if(pNode.right!=null){
            pNode = pNode.right;
            while(pNode.left != null){
                pNode = pNode.left;
            }
            return pNode;
        }
        while(pNode.next!=null){
            if(pNode.next.left==pNode){
                return pNode.next;
            }
            pNode = pNode.next;
        }
        return null;
    }
```
---
### 对称的二叉树
> 请实现一个函数，用来判断一颗二叉树是不是对称的。注意，如果一个二叉树同此二叉树的镜像是同样的，定义其为对称的。
- O:
```
boolean isSymmetrical(TreeNode pRoot){
        if(pRoot == null){
            return true;
        }
        Stack<TreeNode> stack = new Stack<>();
        stack.push(pRoot.left);
        stack.push(pRoot.right);
        while (!stack.empty()) {
            TreeNode right = stack.pop();
            TreeNode left = stack.pop();
            if (right == null && left == null) {
                continue;
            }
            if (right == null || left == null) {
                return false;
            }
            if (left.val != right.val) {
               return false;
            }
            stack.push(right.right);
            stack.push(left.left);
            stack.push(left.right);
            stack.push(right.left);
        }
        return true;
    }
```
- P1:
```
boolean isSymmetrical(TreeNode pRoot){
        if(pRoot == null){
            return true;
        }
        return jude(pRoot.left,pRoot.right);
    }
    
     public boolean jude(TreeNode node1, TreeNode node2) {
         if(node1==null&&node2==null){
             return true;
         }else if(node1==null|node2==null){
             return false;
         }
         if(node1.val ==node2.val ){
             return jude(node1.left,node2.right)&&jude(node1.right,node2.left);
         }else{
             return false;
         }
     }
```
---
### 按之字形顺序打印二叉树
> 请实现一个函数按照之字形打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右至左的顺序打印，第三行按照从左到右的顺序打印，其他行以此类推。
- O:
```
public ArrayList<ArrayList<Integer>> Print(TreeNode pRoot) {
        ArrayList<ArrayList<Integer>> arr =new ArrayList<ArrayList<Integer>>();
        if (pRoot == null) {
            return null;
        }
        ArrayList<Integer> list =new ArrayList<>();
        LinkedList<TreeNode> linkedList = new LinkedList<>();
        linkedList.addLast(null); //*
        linkedList.addLast(pRoot);

        boolean leftToRight = true;
        while (linkedList.size() != 1) {
            TreeNode node = linkedList.removeFirst();
            if (node == null) { //在分隔符
                Iterator<TreeNode> iterator = null;
                if (leftToRight) {//左到右
                    iterator = linkedList.iterator();//iterator 自带删除操作
                } else {
                    iterator = linkedList.descendingIterator();
                }
                leftToRight = !leftToRight;
                while (iterator.hasNext()){
                    TreeNode temp = iterator.next();
                    list.add(temp.val);
                }
                arr.add(new ArrayList<Integer>(list));
                list.clear();

                linkedList.addLast(null);
                continue;
            }
            if (node.left!=null){
                linkedList.addLast(node.left);
            }
            if (node.right !=null){
                linkedList.addLast(node.right);
            }
        }
        return arr;
    }
```
---
### 把二叉树打印成多行 
> 从上到下按层打印二叉树，同一层结点从左至右输出。每一层输出一行。
- O:
```
ArrayList<ArrayList<Integer>> print(TreeNode pRoot) {
        ArrayList<ArrayList<Integer>> arr = new ArrayList<ArrayList<Integer>>();
        if (pRoot == null) {
            return arr;
        }

        LinkedList<TreeNode> linkedList = new LinkedList<>();
        linkedList.add(pRoot);

        ArrayList<Integer> list = new ArrayList<>();

        int now = 1, next = 0; // now 代表每行的的个数，当没循环一次，减去一次。直到为0。说明是最后一个元素已经便利完毕。
        while (!linkedList.isEmpty()) {
            now--;
            TreeNode treeNode = linkedList.remove();
            list.add(treeNode.val);
            if (treeNode.left != null) {
                linkedList.add(treeNode.left);
                next++;
            }
            if (treeNode.right != null) {
                linkedList.add(treeNode.right);
                next++;
            }
            if (now ==0){
                arr.add(new ArrayList<>(list));
                list.clear();
                now =next;
                next =0;
            }
        }

        return arr;
    }
```
- P1:
```
ArrayList<ArrayList<Integer> > Print(TreeNode pRoot) {
        ArrayList<ArrayList<Integer>> arr =new ArrayList<ArrayList<Integer>>();
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(pRoot);
        ArrayList<Integer> list =new ArrayList<Integer>();
        list.add(pRoot.val);
        while(!queue.isEmpty()){
            TreeNode treeNode = queue.poll();
       //     ArrayList<Integer> list =new ArrayList<Integer>();
             if (treeNode.left != null) {
                list.add(treeNode.left.val);
                queue.add(treeNode.left);
            }
            if (treeNode.right != null) {
                list.add(treeNode.right.val);
                queue.add(treeNode.right);
            }
            arr.add(new ArrayList<>(list));
            list.clear();
        }
        return arr;
    }
```
---
### 序列化二叉树
> 请实现两个函数，分别用来序列化和反序列化二叉树
二叉树的序列化是指：把一棵二叉树按照某种遍历方式的结果以某种格式保存为字符串，从而使得内存中建立起来的二叉树可以持久保存。序列化可以基于先序、中序、后序、层序的二叉树遍历方式来进行修改，序列化的结果是一个字符串，序列化时通过 某种符号表示空节点（#），以 ！ 表示一个结点值的结束（value!）。
二叉树的反序列化是指：根据某种遍历顺序得到的序列化字符串结果str，重构二叉树。

- O :
```
String Serialize(TreeNode root) {
        if (root == null) {
            return "";
        }
        StringBuilder sb = new StringBuilder();
        return serializeMethod(root, sb);
    }

    public String serializeMethod(TreeNode root, StringBuilder sb) {
        if (root == null) {
            sb.append("#,");
            return sb.toString();
        }
        sb.append(root.val);
        sb.append(',');
        serializeMethod(root.left, sb);
        serializeMethod(root.right, sb);
        return sb.toString();
    }

    TreeNode Deserialize(String str) {
        String[] strs = str.split(",");
        return deSerializeMethod(strs);
    }

    int indexTree = -1;
    TreeNode deSerializeMethod(String[] strs) {
        indexTree++;
        if(strs[indexTree] == ""){
            return null;
        }
        TreeNode treeNode =null;
        if (!strs[indexTree].equals("#")){
            treeNode = new TreeNode(Integer.valueOf(strs[indexTree]));
            treeNode.left = deSerializeMethod(strs);
            treeNode.right = deSerializeMethod(strs);
        }
        return treeNode;
    }
```
---
### 二叉搜索树的第k个结点 
> 给定一棵二叉搜索树，请找出其中的第k小的结点。例如， （5，3，7，2，4，6，8）    中，按结点数值大小顺序第三小结点的值为4。
- O:
```
TreeNode KthNode(TreeNode pRoot, int k){
       if(pRoot==null||k==0){
           return null;
       }
        Stack<TreeNode> stack = new Stack<TreeNode>();
        int count = 0;
        TreeNode node = pRoot;
        do{
            if(node!=null){
                stack.push(node);
                node = node.left;
            }else{
                node = stack.pop();
                count++;
                if(count==k)
                    return node;
                node = node.right;
            }
        }while(node!=null||!stack.isEmpty());
        return null;
    }
```
---
### 数据流中的中位数
> 如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。我们使用Insert()方法读取数据流，使用GetMedian()方法获取当前读取数据的中位数。
- O:
```
 LinkedList<Integer> linkedList = new LinkedList<>();

    public void Insert(Integer num) {
        // 如果为空，或者小于第一个数据
        if (linkedList.size() == 0 || num < linkedList.getFirst()) {
            linkedList.addFirst(num);
        } else {
            boolean isInsert = false;
            // 需要一个一个比较插入了
            for (Integer i : linkedList) {
                if (num < i) {
                    int index = linkedList.indexOf(i);
                    linkedList.add(index, num);
                    isInsert = true;
                    break;
                }
            }
            if (!isInsert) {
                linkedList.addLast(num);
            }
        }
    }

    public Double GetMedian() {
        if (linkedList.size() == 0) {
            return null;
        }
        if (linkedList.size() % 2 == 0) {
            int i = linkedList.size() / 2;
            double temp = (linkedList.get(i - 1) + linkedList.get(i));
            return temp / 2;
        }
        return (double) linkedList.get(linkedList.size() / 2);
    }
```
---
### 滑动窗口的最大值
> 给定一个数组和滑动窗口的大小，找出所有滑动窗口里数值的最大值。例如，如果输入数组{2,3,4,2,6,2,5,1}及滑动窗口的大小3，那么一共存在6个滑动窗口，他们的最大值分别为{4,4,6,6,6,5}； 针对数组{2,3,4,2,6,2,5,1}的滑动窗口有以下6个： {[2,3,4],2,6,2,5,1}， {2,[3,4,2],6,2,5,1}， {2,3,[4,2,6],2,5,1}， {2,3,4,[2,6,2],5,1}， {2,3,4,2,[6,2,5],1}， {2,3,4,2,6,[2,5,1]}。
- O: ArrayDeque
```
 	public ArrayList<Integer> maxInWindows(int[] num, int size) {
        ArrayList<Integer> list;
        ArrayList<Integer> arrayList = new ArrayList<Integer>();
        if (num.length == 0 || size ==0) {
            return arrayList;
        }
        int length = num.length;
        if (length < size) {
            return arrayList;
        } else {
            for (int i = 0; i < num.length - size + 1; i++) {
                list=new ArrayList<Integer>();
                for (int j = i; j < i + size; j++) {
                    list.add(num[j]);
                }
                Collections.sort(list);
                arrayList.add(list.get(list.size()-1));
            }
        }
        return arrayList;
    }
```
---
### 矩阵中的路径
> 请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一个格子开始，每一步可以在矩阵中向左，向右，向上，向下移动一个格子。如果一条路径经过了矩阵中的某一个格子，则该路径不能再进入该格子。 例如 a b c e s f c s a d e e 矩阵中包含一条字符串"bcced"的路径，但是矩阵中不包含"abcb"路径，因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入该格子。
- O:
```
	public boolean hasPath(char[] matrix, int rows, int cols, char[] str) {
        boolean[] flag = new boolean[matrix.length]; //  定义标志数组（是否走过这个数据）
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (find(matrix, i, j, rows, cols, str, flag, 0)){
                    return true;
                }
            }
        }
        return false;
    }

    // matrix 初始矩阵，索引矩阵i，索引矩阵j，矩阵行数，矩阵列数，待判断的字符串，是否走过，字符串索引
    public boolean find(char[] matrix, int i, int j, int rows, int cols, char[] str, boolean[] flag, int k) {
        // 计算当前索引在一维数组中的位置
        int index = i * cols + j;
        //终止条件
        // 矩阵的某个字符和当前比较的不相等||当前标志显示已经访问过
        if (i < 0 || j < 0 || i >= rows || j >= cols || matrix[index] != str[k] || flag[index] == true) {
            return false;
        }
        //判断是否吻合了
        if (k == str.length - 1) {
            return true;
        }
        //要走的第一个位置置为true，表示已经走过了
        flag[index] = true;
        //回溯
        if (find(matrix, i -1, j, rows, cols, str, flag, k + 1) ||
                find(matrix, i + 1, j, rows, cols, str, flag, k + 1) ||
                find(matrix, i, j - 1, rows, cols, str, flag, k + 1) ||
                find(matrix, i, j + 1, rows, cols, str, flag, k + 1)) {
            return true;
        }
        //走到这，说明这一条路不通，还原，再试其他的路径
        flag[index] = false;
        return false;
    }
```
---
### 机器人的运动范围
> 地上有一个m行和n列的方格。一个机器人从坐标0,0的格子开始移动，每一次只能向左，右，上，下四个方向移动一格，但是不能进入行坐标和列坐标的数位之和大于k的格子。 例如，当k为18时，机器人能够进入方格（35,37），因为3+5+3+7 = 18。但是，它不能进入方格（35,38），因为3+5+3+8 = 19。请问该机器人能够达到多少个格子？
- O:
```
	public int movingCount(int threshold, int rows, int cols) {
        boolean[] flag = new boolean[rows * cols];
        return cal(threshold, 0, 0, rows, cols, flag);
    }

    public int cal(int threshold, int i, int j, int rows, int cols, boolean[] flag) {
        if (i < 0 || j < 0 || i >= rows || j >= cols) {
            return 0;
        }
        int index = i * cols + j;
        // 判断是否符合条件
        if (flag[index] || !checkSum(threshold, i, j)) {
            return 0;
        }
        flag[index] = true;
        // 回溯
        return 1 + cal(threshold, i + 1, j, rows, cols, flag) +
                cal(threshold, i - 1, j, rows, cols, flag) +
                cal(threshold, i, j + 1, rows, cols, flag) +
                cal(threshold, i, j - 1, rows, cols, flag);
    }

    private boolean checkSum(int threshold, int row, int col) {
        int sum = 0;
        while (row != 0) {
            sum += row % 10;
            row = row / 10;
        }
        while (col != 0) {
            sum += col % 10;
            col = col / 10;
        }
        if (sum > threshold) {
            return false;
        }
        return true;
    }
```
---
### 剪绳子
> 给你一根长度为n的绳子，请把绳子剪成整数长的m段（m、n都是整数，n>1并且m>1），每段绳子的长度记为k[0],k[1],...,k[m]。请问k[0]xk[1]x...xk[m]可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。
- O:
```
	public int cutRope(int target) {
       int[] dp = new int[target + 1];
        if (target == 2) {
            return 1;
        }
        if (target == 3) {
            return 2;
        }
        dp[0] = 1;
        dp[2] = 2;
        dp[3] = 3;
        int res = 0;//记录最大的
        for (int i = 4; i <= target; i++) {
            for (int j = 1; j <= (i / 2); j++) {// i-j时候已经计算了另一半，所以只需要计算一半即可。
                res = Math.max(res, dp[j] * dp[i - j]);
            }
            dp[i] = res;
        }
        return dp[target];
    }
```
### 贪心算法总结：
总是在对问题求解时，作出看起来是当前是最好的选择。与之相对的是动态规划。
只进不退，贪心。能退能进，线性规划。